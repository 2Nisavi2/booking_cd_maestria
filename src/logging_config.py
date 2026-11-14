"""Configuración centralizada de logging para el proceso de entrenamiento.

Uso en cualquier módulo:

    import logging
    logger = logging.getLogger(__name__)

    from src.logging_config import log_etapa
    with log_etapa(logger, "Etapa 2 - Importancia SHAP"):
        ...
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMATO_CONSOLA = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
FORMATO_ARCHIVO = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


class _ColorFormatter(logging.Formatter):
    """Colorea el nivel en terminales que lo soportan."""

    COLORES = {
        logging.DEBUG: "\033[36m",     # cian
        logging.INFO: "\033[32m",      # verde
        logging.WARNING: "\033[33m",   # amarillo
        logging.ERROR: "\033[31m",     # rojo
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        texto = super().format(record)
        color = self.COLORES.get(record.levelno)
        if color:
            texto = texto.replace(record.levelname, f"{color}{record.levelname}{self.RESET}", 1)
        return texto


def setup_logging(nivel: str = "INFO", archivo: str | Path | None = None,
                  color: bool = True) -> logging.Logger:
    """Configura el logger raíz con salida a consola y, opcionalmente, a archivo.

    Parameters
    ----------
    nivel : nivel mínimo para la consola (DEBUG, INFO, WARNING, ERROR).
    archivo : ruta del archivo de log. El archivo siempre registra desde DEBUG.
    color : colorea los niveles si la salida es una terminal.
    """
    raiz = logging.getLogger()
    raiz.handlers.clear()
    raiz.setLevel(logging.DEBUG)

    consola = logging.StreamHandler(sys.stdout)
    consola.setLevel(getattr(logging, nivel.upper(), logging.INFO))
    usar_color = color and sys.stdout.isatty()
    fmt = _ColorFormatter if usar_color else logging.Formatter
    consola.setFormatter(fmt(FORMATO_CONSOLA, datefmt=FORMATO_FECHA))
    raiz.addHandler(consola)

    if archivo:
        archivo = Path(archivo)
        archivo.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(archivo, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(FORMATO_ARCHIVO, datefmt=FORMATO_FECHA))
        raiz.addHandler(fh)

    # Los warnings de librerías (pandas, sklearn, xgboost) pasan al log
    logging.captureWarnings(True)
    warnings.simplefilter("default")
    # Avisos esperados y sin impacto (categorías raras agrupadas por el OneHotEncoder, versiones)
    for patron in ("Found unknown categories", ".*is deprecated.*", ".*will change in a future version.*"):
        warnings.filterwarnings("ignore", message=patron)
    # Librerías ruidosas en nivel WARNING
    for ruidoso in ("matplotlib", "PIL", "numba", "urllib3", "shap"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    return raiz


def _fmt_duracion(segundos: float) -> str:
    if segundos < 60:
        return f"{segundos:.1f} s"
    minutos, seg = divmod(segundos, 60)
    if minutos < 60:
        return f"{int(minutos)} min {seg:.0f} s"
    horas, minutos = divmod(minutos, 60)
    return f"{int(horas)} h {int(minutos)} min"


@contextmanager
def log_etapa(logger: logging.Logger, nombre: str, separador: bool = False):
    """Registra el inicio, el fin y la duración de una etapa.

    Si la etapa falla, registra el error con traza y vuelve a lanzar la excepción.
    """
    if separador:
        logger.info("=" * 70)
    logger.info("▶ Inicia: %s", nombre)
    inicio = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("✖ Falló: %s (tras %s)", nombre, _fmt_duracion(time.perf_counter() - inicio))
        raise
    else:
        logger.info("✔ Termina: %s (%s)", nombre, _fmt_duracion(time.perf_counter() - inicio))
