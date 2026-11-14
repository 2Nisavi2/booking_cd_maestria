"""Modelo de clasificación: propensión a cancelar (is_canceled).

Flujo de 4 etapas con discriminación (AUC, PR-AUC) y calibración (Brier, LogLoss).
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.features import preparar_base_cancelacion
from src.logging_config import log_etapa
from training import artefactos
from training.importancia import importancia_pareto

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

logger = logging.getLogger(__name__)

TARGET = "is_canceled"
CLF_ARBOLES = ["Random Forest", "XGBoost", "Árbol de decisión"]
BASELINE = "Baseline (mayoritaria)"


def _preprocesador_clf(num, cat, escalar):
    pasos = [SimpleImputer(strategy="median")] + ([StandardScaler()] if escalar else [])
    return ColumnTransformer([
        ("num", make_pipeline(*pasos), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                            sparse_output=False)), cat),
    ])


def _catalogo_clf(random_state, max_filas_svm, factor=1.0):
    n = lambda x: max(10, int(x * factor))  # noqa: E731
    # (estimador, escalar, submuestra_train)
    catalogo = {
        BASELINE: (DummyClassifier(strategy="prior"), False, None),
        "KNN": (KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1), True, None),
        "Árbol de decisión": (DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
                                                     random_state=random_state), False, None),
        "Random Forest": (RandomForestClassifier(n_estimators=n(300), min_samples_leaf=2,
                                                 max_features="sqrt", n_jobs=-1,
                                                 random_state=random_state), False, None),
        "SVM (RBF)": (SVC(kernel="rbf", C=1.0, gamma="scale", probability=False,
                          random_state=random_state), True, max_filas_svm),
    }
    if XGBClassifier is not None:
        catalogo["XGBoost"] = (XGBClassifier(n_estimators=n(600), max_depth=8, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                             eval_metric="logloss", n_jobs=-1,
                                             random_state=random_state), False, None)
    return catalogo


def _entrenar_clf(Xtr, Xte, ytr, yte, etapa, random_state, max_filas_svm, umbral, factor):
    cat = [c for c in Xtr.columns if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num = [c for c in Xtr.columns if c not in cat]
    catalogo = _catalogo_clf(random_state, max_filas_svm, factor)
    if XGBClassifier is None:
        logger.warning("xgboost no está instalado: se omite XGBoost (pip install xgboost).")
    eval_tr = Xtr.sample(min(20_000, len(Xtr)), random_state=random_state)
    modelos, filas, probas = {}, [], {}
    for i, (nombre, (est, escalar, sub)) in enumerate(catalogo.items(), 1):
        inicio = time.time()
        if sub and len(Xtr) > sub:
            Xfit, _, yfit, _ = train_test_split(Xtr, ytr, train_size=sub, stratify=ytr,
                                                random_state=random_state)
            logger.info("[%d/%d] Entrenando %s con submuestra estratificada de %d filas...",
                        i, len(catalogo), nombre, len(Xfit))
        else:
            Xfit, yfit = Xtr, ytr
            logger.info("[%d/%d] Entrenando %s (%d variables)...", i, len(catalogo), nombre, Xtr.shape[1])
        modelo = make_pipeline(_preprocesador_clf(num, cat, escalar), est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(Xfit, yfit)
            if hasattr(modelo, "predict_proba"):
                s_te = modelo.predict_proba(Xte)[:, 1]
                s_tr = modelo.predict_proba(eval_tr)[:, 1]
                pred = (s_te >= umbral).astype(int)
                brier = brier_score_loss(yte, s_te)
                ll = log_loss(yte, np.clip(s_te, 1e-6, 1 - 1e-6))
                probas[nombre] = s_te
            else:  # SVM: puntaje sin probabilidad
                s_te = modelo.decision_function(Xte)
                s_tr = modelo.decision_function(eval_tr)
                pred = modelo.predict(Xte)
                brier = ll = np.nan
        filas.append({"etapa": etapa, "modelo": nombre, "n_variables": Xtr.shape[1],
                      "AUC_train": roc_auc_score(ytr.loc[eval_tr.index], s_tr),
                      "AUC_test": roc_auc_score(yte, s_te),
                      "PR_AUC_test": average_precision_score(yte, s_te),
                      "Brier": brier, "LogLoss": ll,
                      "accuracy": accuracy_score(yte, pred),
                      "precision": precision_score(yte, pred, zero_division=0),
                      "recall": recall_score(yte, pred), "f1": f1_score(yte, pred),
                      "segundos": time.time() - inicio})
        modelos[nombre] = modelo
        f = filas[-1]
        logger.info("      ✓ %s: AUC test = %.3f | Brier = %s | F1 = %.3f (%.0f s)", nombre,
                    f["AUC_test"], "n/a" if np.isnan(brier) else f"{brier:.4f}", f["f1"], f["segundos"])
    metricas = pd.DataFrame(filas).set_index("modelo").sort_values("AUC_test", ascending=False)
    metricas["brecha_AUC"] = metricas["AUC_train"] - metricas["AUC_test"]
    return modelos, metricas.round(4), probas


def ejecutar_clasificadores_iniciales(df, test_size=0.2, random_state=42, max_filas_svm=20_000,
                                      umbral=0.5, factor=1.0):
    d = df.drop_duplicates()
    X = d.drop(columns=[TARGET]).copy()
    y = d[TARGET].astype(int)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    X[cat] = X[cat].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y,
                                          random_state=random_state)
    logger.info("Filas: %d -> %d (duplicados eliminados: %d) | train: %d | test: %d | "
                "tasa de cancelación: %.1f %% | variables: %d", len(df), len(d), len(df) - len(d),
                len(Xtr), len(Xte), 100 * y.mean(), X.shape[1])
    modelos, metricas, probas = _entrenar_clf(Xtr, Xte, ytr, yte, "Inicial", random_state,
                                              max_filas_svm, umbral, factor)
    return {"modelos": modelos, "metricas": metricas, "probas": probas, "X_train": Xtr, "X_test": Xte,
            "y_train": ytr, "y_test": yte, "random_state": random_state,
            "max_filas_svm": max_filas_svm, "umbral": umbral}


def ejecutar_clasificadores_shap(res_inicial, res_shap, factor=1.0):
    variables = res_shap["variables"]
    logger.info("Variables: %d -> %d", res_inicial["X_train"].shape[1], len(variables))
    modelos, metricas, probas = _entrenar_clf(
        res_inicial["X_train"][variables], res_inicial["X_test"][variables],
        res_inicial["y_train"], res_inicial["y_test"], f"SHAP {res_shap['umbral']:.0%}",
        res_inicial["random_state"], res_inicial["max_filas_svm"], res_inicial["umbral"], factor)
    return {"modelos": modelos, "metricas": metricas, "probas": probas, "variables": variables}


def elegir_clasificador_campeon(res_inicial, res_post_shap, criterio="AUC_test", tolerancia=0.01):
    menor_mejor = criterio in ("Brier", "LogLoss")
    tabla = pd.concat([res_inicial["metricas"], res_post_shap["metricas"]]).reset_index()
    candidatos = tabla[tabla["modelo"] != BASELINE].dropna(subset=[criterio])
    # El tablero necesita probabilidades: se excluyen modelos sin predict_proba (SVM)
    candidatos = candidatos.dropna(subset=["Brier"])
    mejor = candidatos[criterio].min() if menor_mejor else candidatos[criterio].max()
    limite = mejor * (1 + tolerancia) if menor_mejor else mejor * (1 - tolerancia)
    empatados = candidatos[candidatos[criterio] <= limite] if menor_mejor \
        else candidatos[candidatos[criterio] >= limite]
    campeon = empatados.sort_values(["n_variables", criterio], ascending=[True, menor_mejor]).iloc[0]
    es_inicial = campeon["etapa"] == "Inicial"
    fuente = res_inicial if es_inicial else res_post_shap
    variables = list(res_inicial["X_train"].columns) if es_inicial else res_post_shap["variables"]
    tabla["campeon"] = (tabla["etapa"] == campeon["etapa"]) & (tabla["modelo"] == campeon["modelo"])
    tabla = tabla.sort_values(criterio, ascending=menor_mejor).reset_index(drop=True)
    justificacion = (f"Campeón: {campeon['modelo']} ({campeon['etapa']}, {int(campeon['n_variables'])} "
                     f"variables) con {criterio} = {campeon[criterio]:.4f} | Brier = "
                     f"{campeon['Brier']:.4f}. Mejor valor observado: {mejor:.4f}; {len(empatados)} "
                     f"modelo(s) dentro de la tolerancia del {tolerancia:.0%}, se elige el de menos variables.")
    logger.info("🏆 %s", justificacion)
    proba = fuente["probas"][campeon["modelo"]]
    return {"tabla": tabla, "campeon": campeon, "modelo": fuente["modelos"][campeon["modelo"]],
            "variables": variables, "justificacion": justificacion, "proba_test": proba,
            "probas_etapa": fuente["probas"]}


def ejecutar(df_new: pd.DataFrame, cfg: dict) -> dict:
    """Ejecuta el flujo completo de clasificación y exporta los artefactos del campeón."""
    c, g = cfg["cancelacion"], cfg["general"]
    salida = g["artefactos"] / "cancelacion"
    factor = g["factor_arboles"]

    with log_etapa(logger, "Preparación de la base de cancelación"):
        base = preparar_base_cancelacion(df_new)
        logger.info("Base: %d filas x %d columnas (excluidas fuga, fechas y redundantes)", *base.shape)

    with log_etapa(logger, "Etapa 1/4 - Clasificadores iniciales (cancelación)"):
        ini = ejecutar_clasificadores_iniciales(base, c["test_size"], g["semilla"], c["max_filas_svm"],
                                                c["umbral_decision"], factor)

    with log_etapa(logger, "Etapa 2/4 - Importancia SHAP y selección Pareto (cancelación)"):
        mejor_arbol = [m for m in ini["metricas"].index if m in CLF_ARBOLES][0]
        logger.info("Modelo base para SHAP: %s (mejor modelo de árboles por AUC)", mejor_arbol)
        res_shap = importancia_pareto(ini["modelos"][mejor_arbol], ini["X_train"], ini["y_train"],
                                      mejor_arbol, True, "clasificacion", c["umbral_shap"],
                                      c["n_muestra_shap"], g["semilla"])

    with log_etapa(logger, "Etapa 3/4 - Clasificadores con variables seleccionadas (cancelación)"):
        post = ejecutar_clasificadores_shap(ini, res_shap, factor)

    with log_etapa(logger, "Etapa 4/4 - Elección del clasificador campeón (cancelación)"):
        camp = elegir_clasificador_campeon(ini, post, c["criterio"], c["tolerancia"])
        yte, proba = ini["y_test"].values, camp["proba_test"]
        pred = (proba >= c["umbral_decision"]).astype(int)
        cm = confusion_matrix(yte, pred)
        logger.info("Matriz de confusión (test, umbral %.2f): VN=%d FP=%d FN=%d VP=%d",
                    c["umbral_decision"], cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])

    with log_etapa(logger, "Exportación de artefactos (cancelación)"):
        artefactos.guardar_modelo(camp["modelo"], salida / "campeon.joblib")
        artefactos.guardar_tabla(camp["tabla"], salida / "comparacion_modelos.csv")
        artefactos.guardar_tabla(res_shap["tabla"], salida / "importancia_shap.csv")
        artefactos.guardar_valores_shap(res_shap, salida / "shap_muestra.csv")
        # Probabilidades de test del campeón: el tablero recalcula métricas con cualquier umbral
        artefactos.guardar_tabla(pd.DataFrame({"real": yte, "proba": np.round(proba, 5)}),
                                 salida / "probabilidades_test.csv")
        # Curvas ROC de los modelos de la etapa del campeón
        filas = []
        for nombre, p in camp["probas_etapa"].items():
            if nombre == BASELINE:
                continue
            fpr, tpr, _ = roc_curve(yte, p)
            idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
            filas.append(pd.DataFrame({"modelo": nombre, "fpr": fpr[idx], "tpr": tpr[idx],
                                       "auc": roc_auc_score(yte, p)}))
        artefactos.guardar_tabla(pd.concat(filas), salida / "curvas_roc.csv")
        frac, media = calibration_curve(yte, proba, n_bins=10, strategy="quantile")
        artefactos.guardar_tabla(pd.DataFrame({"proba_media": media, "fraccion_real": frac}),
                                 salida / "calibracion.csv")
        campeon = camp["campeon"]
        base_brier = ini["metricas"].loc[BASELINE, "Brier"]
        meta = artefactos.metadata_modelo(
            tarea="cancelacion", tipo="clasificacion", target=TARGET,
            X=ini["X_train"][camp["variables"]], campeon=campeon, justificacion=camp["justificacion"],
            metodo_importancia=res_shap["metodo"], modelo_base_importancia=res_shap["modelo_base"],
            metricas={k: float(campeon[k]) for k in ["AUC_test", "PR_AUC_test", "Brier", "LogLoss",
                                                      "accuracy", "precision", "recall", "f1"]}
            | {"Brier_baseline": float(base_brier), "tasa_cancelacion_test": float(yte.mean()),
               "n_train": len(ini["X_train"]), "n_test": len(ini["X_test"])},
            cfg=cfg, extra={"umbral_decision": c["umbral_decision"]})
        artefactos.guardar_json(meta, salida / "metadata.json")
    return {"tarea": "cancelacion", "campeon": campeon["modelo"], "etapa": campeon["etapa"],
            "n_variables": int(campeon["n_variables"]), "AUC_test": round(float(campeon["AUC_test"]), 3),
            "Brier": round(float(campeon["Brier"]), 4)}
