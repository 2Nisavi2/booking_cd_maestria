"""Exportación de artefactos que consume el tablero (modelos, tablas y metadatos)."""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _rel(ruta: Path) -> str:
    from src.config import RAIZ
    try:
        return str(Path(ruta).resolve().relative_to(RAIZ))
    except ValueError:
        return str(ruta)

LIMITE_GITHUB_MB = 95  # GitHub rechaza archivos de más de 100 MB


def _mb(ruta: Path) -> float:
    return ruta.stat().st_size / 1e6


def guardar_modelo(modelo, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta, compress=3)
    tam = _mb(ruta)
    logger.info("Modelo guardado: %s (%.1f MB)", _rel(ruta), tam)
    if tam > LIMITE_GITHUB_MB:
        logger.warning("El modelo pesa %.0f MB: supera el límite de GitHub (100 MB). "
                       "Usa Git LFS o un modelo más liviano (p. ej. XGBoost).", tam)


def guardar_tabla(df: pd.DataFrame, ruta: Path, index: bool = False) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=index)
    logger.info("Tabla guardada: %s (%d filas)", _rel(ruta), len(df))


def _convertir(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"No serializable: {type(o)}")


def guardar_json(obj: dict, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_convertir)
    logger.info("JSON guardado: %s", _rel(ruta))


def guardar_valores_shap(res_shap: dict, ruta: Path, max_variables: int = 15) -> None:
    """Guarda una muestra de valores SHAP (formato largo) para graficar la dirección del efecto."""
    sv = res_shap.get("valores_muestra")
    if sv is None:
        logger.info("Sin valores SHAP individuales (se usó permutación): no se exporta %s", ruta.name)
        return
    top = res_shap["tabla"]["variable"].head(max_variables).tolist()
    muestra = res_shap["muestra"]
    filas = []
    for v in top:
        x = muestra[v]
        if pd.api.types.is_numeric_dtype(x):
            rango = x.max() - x.min()
            norm = (x - x.min()) / rango if rango > 0 else x * 0
        else:
            norm = pd.Series(np.nan, index=x.index)
        filas.append(pd.DataFrame({"variable": v, "shap": sv[v].values,
                                   "valor": x.astype(str).values, "valor_norm": norm.values}))
    guardar_tabla(pd.concat(filas, ignore_index=True), ruta)


def _especificacion_variables(X: pd.DataFrame, max_categorias: int = 60) -> dict:
    """Tipo, rango y categorías de cada variable del campeón (para el formulario del tablero)."""
    spec = {}
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            spec[c] = {"tipo": "numerica", "min": float(s.min()), "max": float(s.max()),
                       "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                       "mediana": float(s.median())}
        else:
            frec = s.value_counts()
            spec[c] = {"tipo": "categorica", "categorias": frec.index[:max_categorias].tolist(),
                       "moda": frec.index[0]}
    return spec


def _versiones() -> dict:
    paquetes = ["scikit-learn", "xgboost", "pandas", "numpy", "shap", "joblib"]
    v = {"python": platform.python_version()}
    for p in paquetes:
        try:
            v[p] = importlib_metadata.version(p)
        except importlib_metadata.PackageNotFoundError:
            v[p] = None
    return v


def metadata_modelo(tarea, tipo, target, X, campeon, justificacion, metodo_importancia,
                    modelo_base_importancia, metricas, cfg, extra: dict | None = None) -> dict:
    meta = {
        "tarea": tarea, "tipo": tipo, "target": target,
        "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
        "modo_rapido": bool(cfg["general"]["rapido"]),
        "semilla": cfg["general"]["semilla"],
        "campeon": {"modelo": campeon["modelo"], "etapa": campeon["etapa"],
                    "n_variables": int(campeon["n_variables"])},
        "justificacion": justificacion,
        "importancia": {"metodo": metodo_importancia, "modelo_base": modelo_base_importancia},
        "variables": list(X.columns),
        "especificacion": _especificacion_variables(X),
        "metricas": metricas,
        "versiones": _versiones(),
    }
    if extra:
        meta.update(extra)
    return meta
