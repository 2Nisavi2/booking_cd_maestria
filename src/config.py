"""Carga de la configuración (config.yaml) con ajustes del modo rápido."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]


def cargar_config(ruta: str | Path = RAIZ / "config.yaml", rapido: bool = False) -> dict:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {ruta}")
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["general"]["rapido"] = rapido
    cfg["general"]["factor_arboles"] = 1.0

    if rapido:
        r = cfg.get("rapido", {})
        cfg["general"]["factor_arboles"] = r.get("factor_arboles", 0.2)
        for tarea in ("lead_time", "cancelacion"):
            cfg[tarea]["n_muestra_shap"] = r.get("n_muestra_shap", 500)
        cfg["cancelacion"]["max_filas_svm"] = r.get("max_filas_svm", 5000)
        cfg["segmentacion"]["n_muestra"] = r.get("n_muestra_segmentacion", 3000)
        cfg["segmentacion"]["n_silueta"] = r.get("n_silueta", 3000)
        logger.warning("Modo RÁPIDO activado: menos árboles y muestras más pequeñas "
                       "(solo para pruebas; no publicar estos modelos).")

    # Rutas relativas a la raíz del repositorio
    for clave in ("datos", "artefactos", "logs"):
        p = Path(cfg["general"][clave])
        cfg["general"][clave] = p if p.is_absolute() else RAIZ / p
    return cfg
