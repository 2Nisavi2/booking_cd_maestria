"""Agregados del análisis descriptivo y de calidad de datos para el storytelling del tablero.

Precalcular estas tablas evita procesar las 119 mil reservas en cada visita al tablero.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.features import MESES, MESES_ES
from src.logging_config import log_etapa
from training import artefactos

logger = logging.getLogger(__name__)


def _cramer_v(x: pd.Series, y: pd.Series) -> float:
    tabla = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(tabla)[0]
    return float(np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1))))


def _auc_univariado(x: pd.Series, y: pd.Series, semilla: int) -> float:
    """AUC de una sola variable. En categóricas usa target encoding fuera de pliegue."""
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > 15:
        auc = roc_auc_score(y, x.fillna(x.median()))
        return float(max(auc, 1 - auc))
    x = x.astype(str)
    score = pd.Series(np.nan, index=x.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=semilla).split(x, y):
        tasas = y.iloc[tr].groupby(x.iloc[tr]).mean()
        score.iloc[te] = x.iloc[te].map(tasas).fillna(y.iloc[tr].mean()).values
    auc = roc_auc_score(y, score)
    return float(max(auc, 1 - auc))


def _tasa(df, mascara, etiqueta, grupo):
    sub = df.loc[mascara, "is_canceled"]
    return {"factor": etiqueta, "grupo": grupo, "tasa_cancelacion": 100 * sub.mean(), "n": int(len(sub))}


def ejecutar(df: pd.DataFrame, df_new: pd.DataFrame, diag_faltantes: dict, imputacion: pd.DataFrame,
             cfg: dict) -> dict:
    salida = cfg["general"]["artefactos"] / "eda"
    semilla = cfg["general"]["semilla"]
    orden_mes = {m: i for i, m in enumerate(MESES)}
    traduce_mes = dict(zip(MESES, MESES_ES))

    with log_etapa(logger, "EDA - Indicadores generales"):
        n_dup = int(df.duplicated().sum())
        kpis = {
            "n_reservas": len(df), "n_variables": df.shape[1],
            "tasa_cancelacion": float(df["is_canceled"].mean() * 100),
            "tasa_cancelacion_sin_duplicados": float(df.drop_duplicates()["is_canceled"].mean() * 100),
            "n_duplicados": n_dup, "pct_duplicados": 100 * n_dup / len(df),
            "adr_mediana": float(df["adr"].median()), "lead_time_mediana": float(df["lead_time"].median()),
            "lead_time_media": float(df["lead_time"].mean()),
            "pct_agencias": float((df["distribution_channel"] == "TA/TO").mean() * 100),
            "pct_repetidos": float(df["is_repeated_guest"].mean() * 100),
            "pct_city": float((df["hotel"] == "City Hotel").mean() * 100),
            "cancelacion_hotel": (df.groupby("hotel")["is_canceled"].mean() * 100).to_dict(),
            "periodo": "julio 2015 – agosto 2017",
        }
        artefactos.guardar_json(kpis, salida / "kpis.json")
        for k in ("n_reservas", "tasa_cancelacion", "n_duplicados", "adr_mediana", "lead_time_mediana"):
            logger.info("   %-22s %s", k, round(kpis[k], 2))

    with log_etapa(logger, "EDA - Demanda, tarifa y cancelación por mes y hotel"):
        g = df.groupby(["hotel", "arrival_date_month"]).agg(
            reservas=("is_canceled", "size"), tasa_cancelacion=("is_canceled", "mean"),
            adr_mediana=("adr", "median"), lead_time_mediana=("lead_time", "median")).reset_index()
        g["tasa_cancelacion"] *= 100
        g["mes_num"] = g["arrival_date_month"].map(orden_mes) + 1
        g["mes"] = g["arrival_date_month"].map(traduce_mes)
        artefactos.guardar_tabla(g.sort_values(["hotel", "mes_num"]), salida / "mensual_hotel.csv")
        anio = df.groupby("arrival_date_year").size().rename("reservas").reset_index()
        artefactos.guardar_tabla(anio, salida / "reservas_anio.csv")

    with log_etapa(logger, "EDA - Composición de las reservas"):
        filas = []
        for v, etiqueta in [("hotel", "Hotel"), ("market_segment", "Segmento"),
                            ("distribution_channel", "Canal"), ("deposit_type", "Depósito"),
                            ("customer_type", "Tipo de cliente"), ("meal", "Comida")]:
            serie = df[v].replace({"Undefined": "SC"}) if v == "meal" else df[v]
            pct = serie.value_counts(normalize=True) * 100
            principales = pct[pct >= 5]
            for cat, p in principales.items():
                filas.append({"variable": etiqueta, "categoria": cat, "pct": p})
            if pct[pct < 5].sum() > 0:
                filas.append({"variable": etiqueta, "categoria": "Otros", "pct": pct[pct < 5].sum()})
        artefactos.guardar_tabla(pd.DataFrame(filas), salida / "composicion.csv")
        bins = list(range(0, 740, 10))
        hist = pd.cut(df["lead_time"], bins=bins + [800], right=False).value_counts().sort_index()
        artefactos.guardar_tabla(pd.DataFrame({"desde": bins, "reservas": hist.values}),
                                 salida / "lead_time_hist.csv")

    with log_etapa(logger, "EDA - Tasa de cancelación por factor"):
        d = df
        filas = [
            _tasa(d, d["deposit_type"] == "Non Refund", "Depósito no reembolsable", "riesgo"),
            _tasa(d, d["previous_cancellations"] > 0, "Con cancelaciones previas", "riesgo"),
            _tasa(d, d["lead_time"] > 365, "Antelación > 1 año", "riesgo"),
            _tasa(d, d["days_in_waiting_list"] > 0, "Estuvo en lista de espera", "riesgo"),
            _tasa(d, d["market_segment"] == "Groups", "Segmento Groups", "riesgo"),
            _tasa(d, d["total_of_special_requests"] == 0, "Sin solicitudes especiales", "riesgo"),
            _tasa(d, d["hotel"] == "City Hotel", "City Hotel", "riesgo"),
            _tasa(d, d["distribution_channel"] == "TA/TO", "Canal agencias (TA/TO)", "riesgo"),
            _tasa(d, d["customer_type"] == "Transient", "Cliente Transient", "riesgo"),
            _tasa(d, d["country"] == "PRT", "Reserva desde Portugal", "riesgo"),
            _tasa(d, d["hotel"] == "Resort Hotel", "Resort Hotel", "protector"),
            _tasa(d, d["distribution_channel"] == "Direct", "Canal directo", "protector"),
            _tasa(d, d["booking_changes"] > 0, "Reserva modificada", "protector"),
            _tasa(d, d["is_repeated_guest"] == 1, "Huésped repetido", "protector"),
            _tasa(d, d["total_of_special_requests"] > 0, "Con solicitudes especiales", "protector"),
            _tasa(d, d["lead_time"] < 7, "Antelación < 7 días", "protector"),
            _tasa(d, d["required_car_parking_spaces"] > 0, "Solicita parqueadero", "protector"),
        ]
        tabla = pd.DataFrame(filas).sort_values("tasa_cancelacion", ascending=False)
        artefactos.guardar_tabla(tabla, salida / "cancelacion_factores.csv")

    with log_etapa(logger, "EDA - Asociación de variables con la cancelación"):
        y = df["is_canceled"]
        filas = []
        for v in ["deposit_type", "market_segment", "distribution_channel", "hotel", "customer_type",
                  "meal", "reserved_room_type"]:
            filas.append({"variable": v, "asociacion": _cramer_v(df[v], y), "medida": "V de Cramér"})
        for v in ["lead_time", "total_of_special_requests", "required_car_parking_spaces",
                  "booking_changes", "previous_cancellations", "is_repeated_guest", "adr",
                  "days_in_waiting_list", "adults"]:
            filas.append({"variable": v, "asociacion": float(df[v].corr(y)), "medida": "Pearson"})
        tabla = pd.DataFrame(filas)
        tabla = tabla.reindex(tabla["asociacion"].abs().sort_values(ascending=False).index)
        artefactos.guardar_tabla(tabla, salida / "asociacion_cancelacion.csv")

    with log_etapa(logger, "EDA - Antelación y tarifa por grupo"):
        filas = []
        for v, etiqueta in [("market_segment", "Segmento"), ("deposit_type", "Depósito"),
                            ("customer_type", "Tipo de cliente")]:
            g = df.groupby(v).agg(lead_time_mediana=("lead_time", "median"),
                                  adr_mediana=("adr", "median"), n=("adr", "size")).reset_index()
            g = g[g["n"] >= 100].rename(columns={v: "categoria"})
            g["variable"] = etiqueta
            filas.append(g)
        g = df.groupby("is_canceled")["lead_time"].median().reset_index()
        filas.append(pd.DataFrame({"categoria": g["is_canceled"].map({0: "Efectiva", 1: "Cancelada"}),
                                   "lead_time_mediana": g["lead_time"], "variable": "Resultado"}))
        artefactos.guardar_tabla(pd.concat(filas, ignore_index=True), salida / "grupos_lead_adr.csv")

    with log_etapa(logger, "EDA - Calidad de datos y fuga de información"):
        artefactos.guardar_tabla(diag_faltantes["resumen"].reset_index(names="variable")
                                 .query("n_faltantes > 0"), salida / "faltantes.csv")
        artefactos.guardar_tabla(diag_faltantes["diagnostico"], salida / "diagnostico_faltantes.csv")
        artefactos.guardar_tabla(imputacion, salida / "imputacion.csv")
        motivos = {
            "reservation_status": "Es el resultado de la reserva (Canceled / Check-Out)",
            "reservation_status_date": "Se registra al cerrar la reserva",
            "assigned_room_type": "Se conoce al llegar el huésped",
            "room_changed": "Se conoce al llegar el huésped",
            "needs_parking": "Probablemente se registra al llegar (0 % de cancelación)",
            "required_car_parking_spaces": "Probablemente se registra al llegar (0 % de cancelación)",
        }
        yb = df_new["is_canceled"]
        filas = []
        for v in list(motivos) + ["agent", "country", "lead_time", "deposit_type",
                                  "total_of_special_requests", "market_segment", "previous_cancel_rate",
                                  "customer_type", "adr"]:
            x = df_new[v].astype(str) if v == "agent" else df_new[v]
            auc = _auc_univariado(x, yb, semilla)
            filas.append({"variable": v, "auc_univariado": round(auc, 3),
                          "accion": "Excluida" if v in motivos else "Conservada",
                          "motivo": motivos.get(v, "Predictor legítimo (se conoce al reservar)")})
            logger.info("   AUC univariado %-28s %.3f  [%s]", v, auc, filas[-1]["accion"])
        artefactos.guardar_tabla(pd.DataFrame(filas), salida / "fuga.csv")

    with log_etapa(logger, "EDA - Opciones de los formularios del tablero"):
        def top(col, n):
            return df[col].dropna().value_counts().index[:n].tolist()
        tasa_pais = df.groupby("country")["is_canceled"].agg(["mean", "size"])
        opciones = {
            "hotel": top("hotel", 2), "meal": [m for m in top("meal", 5) if m != "Undefined"],
            "market_segment": [m for m in top("market_segment", 8) if m != "Undefined"],
            "distribution_channel": [m for m in top("distribution_channel", 5) if m != "Undefined"],
            "customer_type": top("customer_type", 4), "deposit_type": top("deposit_type", 3),
            "reserved_room_type": sorted(top("reserved_room_type", 10)),
            "country": top("country", 40),
            "agent": [int(a) for a in df["agent"].dropna().value_counts().index[:40]],
            "tasa_pais": {k: round(float(v), 3) for k, v in tasa_pais.loc[tasa_pais["size"] >= 100, "mean"].items()},
        }
        artefactos.guardar_json(opciones, salida / "opciones_formulario.json")
    return {"tarea": "eda", "tablas": len(list(salida.glob("*")))}
