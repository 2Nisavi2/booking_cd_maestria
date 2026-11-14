"""Modelo de regresión: anticipación de la reserva (lead_time, en días).

Flujo de 4 etapas (notebook Entrega III):
1. Modelos iniciales con todas las variables.
2. Importancia SHAP sobre el mejor modelo de árboles y selección Pareto.
3. Reentrenamiento con las variables seleccionadas (mismo split).
4. Elección del campeón por MAE_test con tolerancia (parsimonia).
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import preparar_base_lead_time
from src.logging_config import log_etapa
from training import artefactos
from training.importancia import importancia_pareto

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

logger = logging.getLogger(__name__)

TARGET = "lead_time"
MODELOS_ARBOLES = ["Random Forest", "Gradient Boosting", "XGBoost"]


def _preprocesador(num, cat, escalar):
    pasos_num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if escalar else [])
    return ColumnTransformer([
        ("num", make_pipeline(*pasos_num), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                            drop="first", sparse_output=False)), cat),
    ])


def _catalogo_modelos(random_state, factor=1.0):
    n = lambda x: max(10, int(x * factor))  # noqa: E731
    cv = KFold(5, shuffle=True, random_state=random_state)
    catalogo = {
        "Baseline (media)": (DummyRegressor(), False),
        "Ridge": (RidgeCV(alphas=np.logspace(-3, 3, 25)), True),
        "Lasso": (LassoCV(alphas=np.logspace(-4, 2, 49), cv=cv, max_iter=5000,
                          random_state=random_state), True),
        "ElasticNet": (ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.99], alphas=np.logspace(-4, 2, 30),
                                    cv=cv, max_iter=5000, random_state=random_state), True),
        "Random Forest": (RandomForestRegressor(n_estimators=n(150), min_samples_leaf=3,
                                                max_features=0.4, n_jobs=-1,
                                                random_state=random_state), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=n(300), max_depth=5,
                                                        learning_rate=0.1, subsample=0.8,
                                                        random_state=random_state), False),
    }
    if XGBRegressor is not None:
        catalogo["XGBoost"] = (XGBRegressor(n_estimators=n(600), max_depth=8, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                            n_jobs=-1, random_state=random_state), False)
    return catalogo


def _entrenar(Xtr, Xte, ytr, yte, etapa, random_state, factor):
    cat = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num = [c for c in Xtr.columns if c not in cat]
    catalogo = _catalogo_modelos(random_state, factor)
    if XGBRegressor is None:
        logger.warning("xgboost no está instalado: se omite XGBoost (pip install xgboost).")
    modelos, filas = {}, []
    for i, (nombre, (reg, escalar)) in enumerate(catalogo.items(), 1):
        logger.info("[%d/%d] Entrenando %s (%d variables)...", i, len(catalogo), nombre, Xtr.shape[1])
        inicio = time.time()
        modelo = make_pipeline(_preprocesador(num, cat, escalar), reg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(Xtr, ytr)
        p_tr = np.clip(modelo.predict(Xtr), 0, None)  # los días no pueden ser negativos
        p_te = np.clip(modelo.predict(Xte), 0, None)
        filas.append({"etapa": etapa, "modelo": nombre, "n_variables": Xtr.shape[1],
                      "R2_train": r2_score(ytr, p_tr), "R2_test": r2_score(yte, p_te),
                      "MAE_train": mean_absolute_error(ytr, p_tr),
                      "MAE_test": mean_absolute_error(yte, p_te),
                      "RMSE_test": mean_squared_error(yte, p_te) ** 0.5,
                      "segundos": time.time() - inicio})
        modelos[nombre] = modelo
        f = filas[-1]
        logger.info("      ✓ %s: R2 test = %.3f | MAE test = %.1f días | RMSE = %.1f (%.0f s)",
                    nombre, f["R2_test"], f["MAE_test"], f["RMSE_test"], f["segundos"])
    metricas = pd.DataFrame(filas).set_index("modelo").sort_values("MAE_test")
    metricas["brecha_R2"] = metricas["R2_train"] - metricas["R2_test"]
    return modelos, metricas.round(4)


def ejecutar_modelos_iniciales(df, test_size=0.2, random_state=42, factor=1.0):
    d = df.drop_duplicates()
    X = d.drop(columns=[TARGET]).copy()
    y = d[TARGET].astype(float)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X[cat] = X[cat].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info("Filas: %d -> %d (duplicados eliminados: %d) | train: %d | test: %d | variables: %d",
                len(df), len(d), len(df) - len(d), len(Xtr), len(Xte), X.shape[1])
    modelos, metricas = _entrenar(Xtr, Xte, ytr, yte, "Inicial", random_state, factor)
    return {"modelos": modelos, "metricas": metricas, "X_train": Xtr, "X_test": Xte,
            "y_train": ytr, "y_test": yte, "random_state": random_state}


def ejecutar_modelos_shap(res_inicial, res_shap, factor=1.0):
    variables = res_shap["variables"]
    logger.info("Variables: %d -> %d", res_inicial["X_train"].shape[1], len(variables))
    modelos, metricas = _entrenar(res_inicial["X_train"][variables], res_inicial["X_test"][variables],
                                  res_inicial["y_train"], res_inicial["y_test"],
                                  f"SHAP {res_shap['umbral']:.0%}", res_inicial["random_state"], factor)
    return {"modelos": modelos, "metricas": metricas, "variables": variables}


def elegir_modelo_campeon(res_inicial, res_post_shap, criterio="MAE_test", tolerancia=0.02):
    mayor_mejor = criterio == "R2_test"
    tabla = pd.concat([res_inicial["metricas"], res_post_shap["metricas"]]).reset_index()
    candidatos = tabla[tabla["modelo"] != "Baseline (media)"].copy()
    mejor = candidatos[criterio].max() if mayor_mejor else candidatos[criterio].min()
    limite = mejor * (1 - tolerancia) if mayor_mejor else mejor * (1 + tolerancia)
    empatados = candidatos[candidatos[criterio] >= limite] if mayor_mejor \
        else candidatos[candidatos[criterio] <= limite]
    campeon = empatados.sort_values(["n_variables", criterio],
                                    ascending=[True, not mayor_mejor]).iloc[0]
    es_inicial = campeon["etapa"] == "Inicial"
    fuente = res_inicial if es_inicial else res_post_shap
    variables = list(res_inicial["X_train"].columns) if es_inicial else res_post_shap["variables"]
    tabla["campeon"] = (tabla["etapa"] == campeon["etapa"]) & (tabla["modelo"] == campeon["modelo"])
    tabla = tabla.sort_values(criterio, ascending=not mayor_mejor).reset_index(drop=True)
    justificacion = (f"Campeón: {campeon['modelo']} ({campeon['etapa']}, {int(campeon['n_variables'])} "
                     f"variables) con {criterio} = {campeon[criterio]:.3f}. Mejor valor observado: "
                     f"{mejor:.3f}; {len(empatados)} modelo(s) dentro de la tolerancia del "
                     f"{tolerancia:.0%}, se elige el de menos variables.")
    logger.info("🏆 %s", justificacion)
    return {"tabla": tabla, "campeon": campeon, "modelo": fuente["modelos"][campeon["modelo"]],
            "variables": variables, "justificacion": justificacion}


def ejecutar(df_new: pd.DataFrame, cfg: dict) -> dict:
    """Ejecuta el flujo completo de regresión y exporta los artefactos del campeón."""
    c, g = cfg["lead_time"], cfg["general"]
    salida = g["artefactos"] / "lead_time"
    factor = g["factor_arboles"]

    with log_etapa(logger, "Preparación de la base de lead_time"):
        base = preparar_base_lead_time(df_new)
        logger.info("Base: %d filas x %d columnas (excluidas fuga, redundantes y fechas)", *base.shape)

    with log_etapa(logger, "Etapa 1/4 - Modelos iniciales (lead_time)"):
        ini = ejecutar_modelos_iniciales(base, c["test_size"], g["semilla"], factor)

    with log_etapa(logger, "Etapa 2/4 - Importancia SHAP y selección Pareto (lead_time)"):
        mejor_arbol = [m for m in ini["metricas"].index if m in MODELOS_ARBOLES][0]
        logger.info("Modelo base para SHAP: %s (mejor modelo de árboles por MAE)", mejor_arbol)
        res_shap = importancia_pareto(ini["modelos"][mejor_arbol], ini["X_train"], ini["y_train"],
                                      mejor_arbol, True, "regresion", c["umbral_shap"],
                                      c["n_muestra_shap"], g["semilla"])

    with log_etapa(logger, "Etapa 3/4 - Modelos con variables seleccionadas (lead_time)"):
        post = ejecutar_modelos_shap(ini, res_shap, factor)

    with log_etapa(logger, "Etapa 4/4 - Elección del modelo campeón (lead_time)"):
        camp = elegir_modelo_campeon(ini, post, c["criterio"], c["tolerancia"])

    with log_etapa(logger, "Exportación de artefactos (lead_time)"):
        Xte = ini["X_test"][camp["variables"]]
        pred = np.clip(camp["modelo"].predict(Xte), 0, None)
        artefactos.guardar_modelo(camp["modelo"], salida / "campeon.joblib")
        artefactos.guardar_tabla(camp["tabla"], salida / "comparacion_modelos.csv")
        artefactos.guardar_tabla(res_shap["tabla"], salida / "importancia_shap.csv")
        artefactos.guardar_valores_shap(res_shap, salida / "shap_muestra.csv")
        muestra_pred = pd.DataFrame({"real": ini["y_test"].values, "predicho": pred}) \
            .sample(min(3000, len(pred)), random_state=g["semilla"])
        artefactos.guardar_tabla(muestra_pred, salida / "predicciones_test.csv")
        campeon = camp["campeon"]
        meta = artefactos.metadata_modelo(
            tarea="lead_time", tipo="regresion", target=TARGET, X=ini["X_train"][camp["variables"]],
            campeon=campeon, justificacion=camp["justificacion"], metodo_importancia=res_shap["metodo"],
            modelo_base_importancia=res_shap["modelo_base"],
            metricas={"R2_test": float(campeon["R2_test"]), "MAE_test": float(campeon["MAE_test"]),
                      "RMSE_test": float(campeon["RMSE_test"]),
                      "MAE_baseline": float(ini["metricas"].loc["Baseline (media)", "MAE_test"]),
                      "n_train": len(ini["X_train"]), "n_test": len(ini["X_test"])},
            cfg=cfg)
        artefactos.guardar_json(meta, salida / "metadata.json")
    return {"tarea": "lead_time", "campeon": campeon["modelo"], "etapa": campeon["etapa"],
            "n_variables": int(campeon["n_variables"]), "MAE_test": round(float(campeon["MAE_test"]), 2),
            "R2_test": round(float(campeon["R2_test"]), 3)}
