"""Punto de entrada por terminal del proceso de entrenamiento.

Ejemplos
--------
    python main.py                                  # todas las tareas
    python main.py --tareas cancelacion lead_time   # solo algunas tareas
    python main.py --rapido --nivel-log DEBUG       # prueba rápida con más detalle
    python main.py --datos /ruta/hotel_bookings.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # sin ventanas gráficas en terminal

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

from src.config import cargar_config  # noqa: E402
from src.logging_config import setup_logging  # noqa: E402

TAREAS = ["eda", "lead_time", "cancelacion", "segmentacion"]


def parsear_argumentos(argv=None):
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="Entrena los modelos del proyecto Booking (anticipación, cancelación y "
                    "segmentación) y exporta los artefactos que usa el tablero.")
    p.add_argument("--tareas", nargs="+", choices=TAREAS + ["todas"], default=["todas"],
                   help="Tareas a ejecutar (por defecto: todas). eda = agregados del storytelling.")
    p.add_argument("--config", default=str(RAIZ / "config.yaml"), help="Ruta del archivo de configuración.")
    p.add_argument("--datos", help="Ruta de hotel_bookings.csv (sobrescribe config.yaml).")
    p.add_argument("--salida", help="Carpeta de artefactos (sobrescribe config.yaml).")
    p.add_argument("--rapido", action="store_true",
                   help="Modo rápido: menos árboles y muestras más pequeñas (solo pruebas).")
    p.add_argument("--nivel-log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Nivel de detalle de los mensajes en la terminal.")
    p.add_argument("--sin-archivo-log", action="store_true", help="No guardar el log en logs/.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parsear_argumentos(argv)
    cfg = cargar_config(args.config, rapido=args.rapido)
    if args.datos:
        cfg["general"]["datos"] = Path(args.datos).resolve()
    if args.salida:
        cfg["general"]["artefactos"] = Path(args.salida).resolve()

    archivo_log = None
    if not args.sin_archivo_log:
        archivo_log = cfg["general"]["logs"] / f"entrenamiento_{datetime.now():%Y%m%d_%H%M%S}.log"
    setup_logging(args.nivel_log, archivo_log)
    logger = logging.getLogger("main")

    tareas = TAREAS if "todas" in args.tareas else [t for t in TAREAS if t in args.tareas]
    logger.info("Proyecto Booking · proceso de entrenamiento")
    logger.info("Datos: %s", cfg["general"]["datos"])
    logger.info("Artefactos: %s", cfg["general"]["artefactos"])
    if archivo_log:
        logger.info("Log completo (nivel DEBUG): %s", archivo_log)
    if args.rapido:
        cfg_log = logging.getLogger("src.config")
        cfg_log.warning("Modo RÁPIDO: los modelos resultantes no deben publicarse.")

    from training.pipeline import ejecutar  # import tardío: el logging ya está configurado
    try:
        ejecutar(cfg, tareas)
    except KeyboardInterrupt:
        logger.error("Proceso interrumpido por el usuario.")
        return 130
    except Exception:  # noqa: BLE001
        logger.critical("El entrenamiento terminó con errores. Revisa el log para el detalle.")
        return 1
    logger.info("Entrenamiento finalizado correctamente.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
