"""Diagnóstico del mecanismo de faltantes (MCAR / MAR / MNAR) e imputación guiada.

Adaptado de las funciones `analizar_mecanismo_faltantes` e
`imputar_segun_diagnostico` del notebook (Entregas II y III). Las salidas por
pantalla se reemplazaron por mensajes de logging.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Diagnóstico
# --------------------------------------------------------------------------
def _em_normal(X, max_iter=100, tol=1e-5):
    """EM normal multivariada, vectorizado por patrón de faltantes."""
    n, p = X.shape
    mask = ~np.isnan(X)
    claves, inv_idx = np.unique(mask, axis=0, return_inverse=True)
    inv_idx = inv_idx.ravel()
    grupos = [(claves[g], np.where(inv_idx == g)[0]) for g in range(len(claves))]
    mu = np.nanmean(X, axis=0)
    sigma = np.cov(np.where(mask, X, mu), rowvar=False) + np.eye(p) * 1e-6
    for _ in range(max_iter):
        mu_old, sigma_old = mu.copy(), sigma.copy()
        X_hat, C = X.copy(), np.zeros((p, p))
        for o, idx in grupos:
            m = ~o
            if not m.any():
                continue
            if o.any():
                inv = np.linalg.pinv(sigma[np.ix_(o, o)])
                s_mo = sigma[np.ix_(m, o)]
                X_hat[np.ix_(idx, m)] = mu[m] + (X[np.ix_(idx, o)] - mu[o]) @ inv @ s_mo.T
                C[np.ix_(m, m)] += len(idx) * (sigma[np.ix_(m, m)] - s_mo @ inv @ s_mo.T)
            else:
                X_hat[np.ix_(idx, m)] = mu[m]
                C[np.ix_(m, m)] += len(idx) * sigma[np.ix_(m, m)]
        mu = X_hat.mean(axis=0)
        d = X_hat - mu
        sigma = (d.T @ d + C) / n + np.eye(p) * 1e-9
        if np.max(np.abs(mu - mu_old)) < tol and np.max(np.abs(sigma - sigma_old)) < tol:
            break
    return mu, sigma


def _test_little(df_num: pd.DataFrame, alpha: float) -> dict:
    X = df_num.to_numpy(dtype=float)
    X = X[~np.isnan(X).all(axis=1)]
    p = X.shape[1]
    mu, sigma = _em_normal(X)
    claves, inv_idx = np.unique(~np.isnan(X), axis=0, return_inverse=True)
    inv_idx = inv_idx.ravel()
    d2, gl = 0.0, 0
    for g, obs in enumerate(claves):
        if not obs.any():
            continue
        Xg = X[inv_idx == g][:, obs]
        dif = Xg.mean(axis=0) - mu[obs]
        d2 += Xg.shape[0] * dif @ np.linalg.pinv(sigma[np.ix_(obs, obs)]) @ dif
        gl += obs.sum()
    gl -= p
    p_val = 1 - stats.chi2.cdf(d2, gl) if gl > 0 else np.nan
    return {"estadistico_chi2": float(d2), "grados_libertad": int(gl), "p_valor": float(p_val),
            "n_patrones": int(len(claves)), "rechaza_MCAR": bool(p_val < alpha)}


def _comparar_grupos(serie: pd.Series, indicador: pd.Series) -> dict:
    datos = pd.DataFrame({"x": serie, "falta": indicador}).dropna(subset=["x"])
    g1 = datos.loc[datos["falta"] == 1, "x"]
    g0 = datos.loc[datos["falta"] == 0, "x"]
    if len(g1) < 3 or len(g0) < 3:
        return {"prueba": "insuficiente", "p_valor": np.nan, "efecto": np.nan}
    if pd.api.types.is_numeric_dtype(serie) and serie.nunique() > 10:
        _, p_t = stats.ttest_ind(g1, g0, equal_var=False)
        _, p_u = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        sd = np.sqrt((g1.var() + g0.var()) / 2)
        return {"prueba": "Welch t / Mann-Whitney", "p_valor": float(min(p_t, p_u)),
                "efecto": float((g1.mean() - g0.mean()) / sd) if sd > 0 else 0.0,
                "medida_efecto": "d de Cohen"}
    tabla = pd.crosstab(datos["x"], datos["falta"])
    if min(tabla.shape) < 2:
        return {"prueba": "insuficiente", "p_valor": np.nan, "efecto": np.nan}
    chi2, p, _, _ = stats.chi2_contingency(tabla)
    return {"prueba": "chi2", "p_valor": float(p),
            "efecto": float(np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))),
            "medida_efecto": "V de Cramér"}


def analizar_mecanismo_faltantes(df: pd.DataFrame, alpha: float = 0.05, umbral_auc: float = 0.60,
                                 umbral_efecto: float = 0.20, min_faltantes: int = 30,
                                 estructurales: list | None = None, excluir: list | None = None,
                                 max_filas_modelo: int = 50_000, random_state: int = 42) -> dict:
    """Diagnostica el mecanismo de falta de cada variable y recomienda un tratamiento."""
    estructurales = [c for c in (estructurales or []) if c in df.columns]
    excluir = [c for c in (excluir or []) if c in df.columns]
    df = df.copy()
    rng = np.random.default_rng(random_state)
    predictoras = df.drop(columns=excluir)

    def matriz_predictoras(col):
        X = predictoras.drop(columns=[col], errors="ignore")
        num = X.select_dtypes(include="number")
        num = num.fillna(num.median())
        cat = X.select_dtypes(exclude="number")
        cat = cat.loc[:, cat.nunique() < 30]
        cat = pd.get_dummies(cat.astype("object").fillna("_NA_"), drop_first=True, dtype=float)
        Xm = pd.concat([num, cat], axis=1)
        return Xm.loc[:, Xm.std() > 0]

    def auc_faltante(col):
        y = df[col].isna().astype(int)
        Xm = matriz_predictoras(col)
        if Xm.shape[1] == 0:
            return np.nan
        if len(y) > max_filas_modelo:  # submuestra estratificada
            idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
            frac = max_filas_modelo / len(y)
            sel = np.concatenate([
                rng.choice(idx1, max(int(len(idx1) * frac), min(len(idx1), 30)), replace=False),
                rng.choice(idx0, max(int(len(idx0) * frac), min(len(idx0), 30)), replace=False)])
            Xm, y = Xm.iloc[sel], y.iloc[sel]
        k = int(min(5, y.sum(), len(y) - y.sum()))
        if k < 2:
            return np.nan
        modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = cross_val_predict(modelo, Xm, y, cv=cv, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))

    resumen = pd.DataFrame({
        "n_faltantes": df.isna().sum(),
        "pct_faltantes": (df.isna().mean() * 100).round(4),
        "tipo": df.dtypes.astype(str),
    }).sort_values("pct_faltantes", ascending=False)
    cols_falt = resumen.index[resumen["n_faltantes"] > 0].tolist()
    logger.info("Columnas con faltantes: %d de %d", len(cols_falt), df.shape[1])
    for c in cols_falt:
        logger.info("   · %-10s %8s faltantes (%.4f %%)", c,
                    f"{int(resumen.loc[c, 'n_faltantes']):,}".replace(",", "."),
                    resumen.loc[c, "pct_faltantes"])
    if not cols_falt:
        return {"resumen": resumen, "little": None, "pruebas_mar": None,
                "diagnostico": pd.DataFrame()}

    cols_eval = [c for c in cols_falt if c not in estructurales]

    # Test de Little (MCAR global) sin variables estructurales
    num = predictoras.drop(columns=estructurales, errors="ignore").select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1]
    little = None
    if num.shape[1] >= 2 and num.isna().any().any():
        try:
            little = _test_little(num, alpha)
            logger.info("Test de Little: chi2 = %.2f | gl = %d | p = %.4g -> %s",
                        little["estadistico_chi2"], little["grados_libertad"], little["p_valor"],
                        "se RECHAZA MCAR" if little["rechaza_MCAR"] else "no se rechaza MCAR")
        except Exception as e:  # noqa: BLE001
            little = {"error": str(e)}
            logger.warning("Test de Little no disponible: %s", e)
    else:
        logger.info("Test de Little: no aplica (sin faltantes numéricos no estructurales suficientes)")

    # Pruebas MAR por pares
    registros = []
    for col in cols_eval:
        ind = df[col].isna().astype(int)
        for otra in predictoras.columns.drop(col, errors="ignore"):
            registros.append({"variable_con_faltantes": col, "variable_comparada": otra,
                              **_comparar_grupos(df[otra], ind)})
    pruebas = pd.DataFrame(registros, columns=["variable_con_faltantes", "variable_comparada",
                                               "prueba", "p_valor", "efecto", "medida_efecto"])
    pruebas["p_ajustado_holm"] = np.nan
    ok = pruebas["p_valor"].notna()
    if ok.any():
        p = pruebas.loc[ok, "p_valor"].to_numpy()
        orden, m = np.argsort(p), len(p)
        adj = np.maximum.accumulate((m - np.arange(m)) * p[orden]).clip(max=1)
        p_adj = np.empty(m)
        p_adj[orden] = adj
        pruebas.loc[ok, "p_ajustado_holm"] = p_adj
    pruebas["significativo"] = (pruebas["p_ajustado_holm"] < alpha) & \
                               (pruebas["efecto"].abs() > umbral_efecto)

    # Diagnóstico por variable
    diag = []
    for col in cols_falt:
        n_falt = int(df[col].isna().sum())
        pct = float(resumen.loc[col, "pct_faltantes"])
        base = {"variable": col, "n_faltantes": n_falt, "pct_faltantes": pct,
                "auc_prediccion_faltante": np.nan, "variables_asociadas": ""}
        if col in estructurales:
            diag.append({**base, "mecanismo_probable": "Estructural (no aplica)",
                         "recomendacion": "El NA significa ausencia: codificar (0 / 'Sin dato') "
                                          "y crear indicador binario."
                                          + (" Con >40% NA, usar solo el indicador." if pct > 40 else "")})
            continue
        if n_falt < min_faltantes:
            diag.append({**base, "mecanismo_probable": f"No evaluable ({n_falt} faltantes)",
                         "recomendacion": "Muy pocos casos: imputar con mediana/moda."})
            continue
        sub = pruebas[pruebas["variable_con_faltantes"] == col]
        relacionadas = (sub.loc[sub["significativo"]]
                        .sort_values("efecto", key=abs, ascending=False)["variable_comparada"].tolist())
        auc = auc_faltante(col)
        evidencia_mar = bool(relacionadas) or (not np.isnan(auc) and auc > umbral_auc)
        mecanismo = "MAR (probable)" if evidencia_mar else "MCAR (compatible)"
        rec = ("Imputación multivariada usando: " + ", ".join(relacionadas[:5])
               if evidencia_mar else "Imputación simple aceptable.")
        diag.append({**base, "auc_prediccion_faltante": round(auc, 3) if not np.isnan(auc) else np.nan,
                     "variables_asociadas": ", ".join(relacionadas[:5]),
                     "mecanismo_probable": mecanismo, "recomendacion": rec})
    diagnostico = pd.DataFrame(diag)
    for _, r in diagnostico.iterrows():
        auc_txt = "" if pd.isna(r["auc_prediccion_faltante"]) else f" | AUC faltante = {r['auc_prediccion_faltante']:.3f}"
        logger.info("Diagnóstico %-10s -> %s%s", r["variable"], r["mecanismo_probable"], auc_txt)
    return {"resumen": resumen, "little": little, "pruebas_mar": pruebas, "diagnostico": diagnostico}


# --------------------------------------------------------------------------
# Imputación
# --------------------------------------------------------------------------
def imputar_segun_diagnostico(df: pd.DataFrame, resultado: dict, excluir: list | None = None,
                              max_clases: int = 30, max_filas_modelo: int = 50_000,
                              random_state: int = 42):
    """Aplica a cada variable el tratamiento correspondiente a su diagnóstico.

    Retorna (df_imputado, conclusiones).
    """
    if "diagnostico" not in resultado or resultado["diagnostico"].empty:
        raise ValueError("'resultado' debe venir de analizar_mecanismo_faltantes con faltantes.")
    excluir = [c for c in (excluir or []) if c in df.columns]
    original = df.copy()
    out = df.copy()
    rng = np.random.default_rng(random_state)
    conclusiones = []

    def es_num(col):
        return pd.api.types.is_numeric_dtype(original[col])

    def valor_central(col):
        s = original[col].dropna()
        v = (s.mode().iloc[0] if s.nunique() < 20 else s.median()) if es_num(col) else s.mode().iloc[0]
        return v.item() if hasattr(v, "item") else v

    def predictores(col, asociadas):
        cols = [c for c in asociadas if c in original.columns and c != col and c not in excluir]
        return cols or [c for c in original.columns if c != col and c not in excluir]

    def preparar_X(cols):
        X = original[cols].copy()
        cats = []
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan)
                cats.append(True)
            else:
                cats.append(False)
        return X.astype(float), cats

    def imputar_mice(col, cols_pred):
        num_pred = [c for c in cols_pred if pd.api.types.is_numeric_dtype(original[c])]
        M = original[[col] + num_pred].astype(float)
        imp = IterativeImputer(max_iter=10, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imputado = imp.fit_transform(M)[:, 0]
        s = original[col]
        obs = s.dropna()
        if (obs % 1 == 0).all():
            imputado = np.round(imputado)
        imputado = np.clip(imputado, obs.min(), obs.max())
        return pd.Series(imputado, index=original.index).where(s.isna(), s), "MICE"

    def imputar_clasificador(col, cols_pred):
        y = original[col]
        X, cats = preparar_X(cols_pred)
        top = y.value_counts().index[:max_clases]
        obs = np.where(y.notna() & y.isin(top))[0]
        falt = np.where(y.isna())[0]
        obs_s = rng.choice(obs, max_filas_modelo, replace=False) if len(obs) > max_filas_modelo else obs
        perm = rng.permutation(obs_s)
        corte = int(len(perm) * 0.8)
        tr, te = perm[:corte], perm[corte:]
        modelo = HistGradientBoostingClassifier(categorical_features=cats, max_iter=150,
                                                random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo.fit(X.iloc[tr], y.iloc[tr])
            acc = accuracy_score(y.iloc[te], modelo.predict(X.iloc[te]))
            base = accuracy_score(y.iloc[te], np.repeat(y.iloc[tr].mode().iloc[0], len(te)))
            modelo.fit(X.iloc[obs_s], y.iloc[obs_s])
            pred = modelo.predict(X.iloc[falt])
        res = y.copy()
        res.iloc[falt] = pred
        return res, f"validación: accuracy={acc:.3f} vs. moda={base:.3f}"

    def comparar(antes, despues):
        if pd.api.types.is_numeric_dtype(despues) and pd.api.types.is_numeric_dtype(antes):
            a, d = antes.dropna(), despues.dropna()
            return f"media {a.mean():.3f} -> {d.mean():.3f}; desv. {a.std():.3f} -> {d.std():.3f}"
        pa = antes.value_counts(normalize=True)
        pd_ = despues.value_counts(normalize=True)
        idx = pd_.index.union(pa.index)
        tvd = 0.5 * (pa.reindex(idx, fill_value=0) - pd_.reindex(idx, fill_value=0)).abs().sum()
        return (f"moda '{pa.index[0]}' {pa.iloc[0] * 100:.1f}% -> {pd_.get(pa.index[0], 0) * 100:.1f}%; "
                f"TVD = {tvd * 100:.2f}%")

    for _, r in resultado["diagnostico"].iterrows():
        col = r["variable"]
        if col not in original.columns or original[col].isna().sum() == 0:
            continue
        mec = str(r["mecanismo_probable"])
        n_na = int(original[col].isna().sum())
        asociadas = [c.strip() for c in str(r.get("variables_asociadas", "")).split(",") if c.strip()]
        num = es_num(col)
        metricas, nuevas = "", []
        if mec.startswith("Estructural"):
            if r["pct_faltantes"] > 40:
                ind = f"{col}_registrado"
                out[ind] = original[col].notna().astype(int)
                out = out.drop(columns=[col])
                metodo = f"Indicador binario '{ind}' (1 = tiene {col}); se elimina la variable original"
            else:
                ind = f"{col}_faltante"
                out[ind] = original[col].isna().astype(int)
                relleno = 0 if num else "Sin dato"
                out[col] = original[col].fillna(relleno)
                metodo = f"Relleno con {relleno!r} + indicador '{ind}'"
            nuevas = [ind]
        elif mec.startswith("No evaluable") or mec.startswith("MCAR"):
            v = valor_central(col)
            out[col] = original[col].fillna(v)
            metodo = f"{'Moda' if (not num or original[col].nunique() < 20) else 'Mediana'} ({v!r})"
        else:  # MAR
            cols_pred = predictores(col, asociadas)
            if num:
                out[col], metricas = imputar_mice(col, cols_pred)
                metodo = f"MICE con {', '.join(cols_pred)}"
            else:
                out[col], metricas = imputar_clasificador(col, cols_pred)
                metodo = f"Clasificador HistGradientBoosting con {', '.join(cols_pred)}"
        cambio = comparar(original[col], out[col]) if col in out.columns else \
            f"{nuevas[0]}: {out[nuevas[0]].mean() * 100:.2f}% con valor 1"
        conclusiones.append({"variable": col, "mecanismo": mec, "n_imputados": n_na,
                             "pct_imputado": round(n_na / len(original) * 100, 4), "metodo": metodo,
                             "variables_nuevas": ", ".join(nuevas), "calidad": metricas,
                             "efecto_en_distribucion": cambio})
        logger.info("Imputación %-10s -> %s %s", col, metodo, f"[{metricas}]" if metricas else "")

    conclusiones = pd.DataFrame(conclusiones)
    pendientes = out.isna().sum()
    pendientes = pendientes[pendientes > 0]
    logger.info("Base imputada: %d filas | columnas %d -> %d | NA restantes: %s",
                len(out), original.shape[1], out.shape[1],
                "ninguno" if pendientes.empty else pendientes.to_dict())
    return out, conclusiones
