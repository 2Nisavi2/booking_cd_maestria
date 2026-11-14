"""Importancia de variables con SHAP (o permutación como respaldo) y selección Pareto.

Compartido por los flujos de regresión y clasificación.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

logger = logging.getLogger(__name__)


def _mapa_variables_originales(nombres: list[str], columnas: list[str]) -> list[str]:
    """Asocia cada columna transformada (p. ej. 'cat__country_PRT') a su variable original."""
    originales = sorted(columnas, key=len, reverse=True)

    def original(nombre):
        base = nombre.split("__", 1)[-1]
        for c in originales:
            if base == c or base.startswith(c + "_"):
                return c
        return base

    return [original(n) for n in nombres]


def importancia_pareto(pipe, X_train: pd.DataFrame, y_train: pd.Series, nombre_modelo: str,
                       es_arbol: bool, tarea: str, umbral: float = 0.90, n_muestra: int = 2000,
                       random_state: int = 42) -> dict:
    """Calcula la importancia media |SHAP| por variable original y selecciona hasta `umbral`.

    - Se calcula sobre una muestra de TRAIN (el test no interviene en la selección).
    - Las dummies del One-Hot se suman fila a fila a su variable original.
    - Si shap no está instalado (o el modelo no es de árboles) usa permutación.

    Retorna: variables, tabla, metodo, umbral, modelo_base y (si hay SHAP) una
    muestra de valores SHAP agregados para graficar la dirección del efecto.
    """
    muestra = X_train.sample(min(n_muestra, len(X_train)), random_state=random_state)
    pre, est = pipe[:-1], pipe[-1]
    valores_muestra = None

    if shap is not None and es_arbol:
        metodo = "SHAP - TreeExplainer"
        logger.info("Calculando valores SHAP (TreeExplainer) sobre '%s' con %d filas de train...",
                    nombre_modelo, len(muestra))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = pre.transform(muestra)
        nombres = list(pre.get_feature_names_out())
        mapa = _mapa_variables_originales(nombres, list(X_train.columns))
        valores = shap.TreeExplainer(est).shap_values(Z, check_additivity=False)
        if isinstance(valores, list):          # versiones antiguas: lista por clase
            valores = valores[1]
        elif np.ndim(valores) == 3:             # versiones nuevas: (filas, variables, clases)
            valores = valores[:, :, 1]
        sv = pd.DataFrame(valores, columns=nombres).T.groupby(mapa).sum().T
        importancia = sv.abs().mean()
        valores_muestra = sv
        valores_muestra.index = muestra.index
    else:
        motivo = "shap no instalado" if shap is None else "modelo no basado en árboles"
        metodo = f"Permutación ({motivo})"
        scoring = "roc_auc" if tarea == "clasificacion" else "neg_mean_absolute_error"
        logger.warning("No se usará SHAP (%s): importancia por permutación con scoring=%s "
                       "sobre %d filas de train...", motivo, scoring, len(muestra))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = permutation_importance(pipe, muestra, y_train.loc[muestra.index], scoring=scoring,
                                        n_repeats=3, random_state=random_state, n_jobs=1)
        importancia = pd.Series(np.clip(pi.importances_mean, 0, None), index=muestra.columns)

    tabla = (importancia.rename("importancia").sort_values(ascending=False).to_frame()
             .rename_axis("variable").reset_index())
    total = tabla["importancia"].sum()
    tabla["importancia_%"] = tabla["importancia"] / total * 100 if total > 0 else 0.0
    tabla["acumulado_%"] = tabla["importancia_%"].cumsum()
    n_sel = min(int((tabla["acumulado_%"] < umbral * 100).sum()) + 1, len(tabla))
    tabla["seleccionada"] = tabla.index < n_sel
    variables = tabla.loc[tabla["seleccionada"], "variable"].tolist()

    logger.info("Método: %s | variables: %d -> %d seleccionadas (acumulan %.1f %% >= %.0f %%)",
                metodo, len(tabla), len(variables), tabla.loc[n_sel - 1, "acumulado_%"], umbral * 100)
    for _, r in tabla.head(10).iterrows():
        logger.info("   %-32s %6.2f %%  (acum. %5.1f %%)%s", r["variable"], r["importancia_%"],
                    r["acumulado_%"], "" if r["seleccionada"] else "  [no seleccionada]")
    if len(tabla) > 10:
        logger.debug("Tabla completa de importancia:\n%s", tabla.round(3).to_string())

    return {"variables": variables, "tabla": tabla, "modelo_base": nombre_modelo,
            "metodo": metodo, "umbral": umbral, "valores_muestra": valores_muestra,
            "muestra": muestra}
