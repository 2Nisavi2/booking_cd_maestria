"""Orquestador del proceso de entrenamiento."""
from __future__ import annotations

import logging
import time
from datetime import datetime

from src.datos import cargar_datos
from src.faltantes import analizar_mecanismo_faltantes, imputar_segun_diagnostico
from src.features import crear_variables_derivadas
from src.logging_config import log_etapa
from training import artefactos

logger = logging.getLogger(__name__)

TAREAS = ["eda", "lead_time", "cancelacion", "segmentacion"]


def preprocesar(cfg: dict):
    """Carga, diagnóstico de faltantes, imputación y variables derivadas (base df_new)."""
    p = cfg["preprocesamiento"]
    with log_etapa(logger, "Carga de datos", separador=True):
        df = cargar_datos(cfg["general"]["datos"])
    with log_etapa(logger, "Diagnóstico del mecanismo de faltantes (MCAR / MAR / MNAR)"):
        diag = analizar_mecanismo_faltantes(df, estructurales=p["estructurales"], excluir=p["excluir"],
                                            random_state=cfg["general"]["semilla"])
    with log_etapa(logger, "Imputación según diagnóstico"):
        df_imp, conclusiones = imputar_segun_diagnostico(df, diag, excluir=p["excluir"],
                                                         random_state=cfg["general"]["semilla"])
    with log_etapa(logger, "Creación de variables derivadas"):
        df_new = crear_variables_derivadas(df_imp)
        logger.info("df_new: %d filas x %d columnas (%d derivadas)", *df_new.shape,
                    df_new.shape[1] - df_imp.shape[1])
    return df, df_new, diag, conclusiones


def ejecutar(cfg: dict, tareas: list[str]) -> dict:
    inicio = time.perf_counter()
    logger.info("Tareas a ejecutar: %s", ", ".join(tareas))
    df, df_new, diag, conclusiones = preprocesar(cfg)
    resultados = []
    for i, tarea in enumerate(tareas, 1):
        with log_etapa(logger, f"TAREA {i}/{len(tareas)}: {tarea.upper()}", separador=True):
            if tarea == "eda":
                from training import eda
                resultados.append(eda.ejecutar(df, df_new, diag, conclusiones, cfg))
            elif tarea == "lead_time":
                from training import regresion
                resultados.append(regresion.ejecutar(df_new, cfg))
            elif tarea == "cancelacion":
                from training import clasificacion
                resultados.append(clasificacion.ejecutar(df_new, cfg))
            elif tarea == "segmentacion":
                from training import segmentacion
                resultados.append(segmentacion.ejecutar(df_new, cfg))
    duracion = time.perf_counter() - inicio
    resumen = {"fecha": datetime.now().isoformat(timespec="seconds"), "duracion_segundos": round(duracion),
               "modo_rapido": cfg["general"]["rapido"], "resultados": resultados}
    artefactos.guardar_json(resumen, cfg["general"]["artefactos"] / "resumen_entrenamiento.json")
    logger.info("=" * 70)
    logger.info("RESUMEN DEL ENTRENAMIENTO (%.1f min)", duracion / 60)
    for r in resultados:
        logger.info("   %s", " | ".join(f"{k}: {v}" for k, v in r.items()))
    return resumen
