"""Acceso a artefactos y lógica de predicción del tablero (sin dependencias de Dash).

Separar esta capa permite probarla con pytest sin levantar el servidor.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.features import MESES, construir_reserva  # noqa: E402

ARTEFACTOS = RAIZ / "artifacts"

# Campo del formulario -> variables del modelo que dependen de él
DEPENDENCIAS = {
    "hotel": ["hotel"],
    "fecha_llegada": ["arrival_date_year", "arrival_date_month", "arrival_date_week_number",
                      "arrival_date_day_of_month", "arrival_weekday", "arrival_is_weekend",
                      "arrival_month_num", "arrival_quarter", "season", "high_season"],
    "lead_time": ["lead_time", "lead_time_cat"],
    "stays_in_weekend_nights": ["stays_in_weekend_nights", "total_nights", "weekend_ratio",
                                "stay_type", "total_revenue"],
    "stays_in_week_nights": ["stays_in_week_nights", "total_nights", "weekend_ratio", "stay_type",
                             "total_revenue"],
    "adults": ["adults", "total_guests", "guest_type", "adr_per_guest"],
    "children": ["children", "has_kids", "total_guests", "guest_type", "adr_per_guest"],
    "babies": ["babies", "has_kids", "total_guests", "guest_type", "adr_per_guest"],
    "meal": ["meal"], "country": ["country", "is_portugal"], "market_segment": ["market_segment"],
    "distribution_channel": ["distribution_channel"], "is_repeated_guest": ["is_repeated_guest"],
    "previous_cancellations": ["previous_cancellations", "previous_cancel_rate",
                               "total_previous_bookings", "has_previous_cancel"],
    "previous_bookings_not_canceled": ["previous_bookings_not_canceled", "previous_cancel_rate",
                                       "total_previous_bookings"],
    "reserved_room_type": ["reserved_room_type"],
    "booking_changes": ["booking_changes", "has_booking_changes"],
    "deposit_type": ["deposit_type", "has_deposit"], "agent": ["agent", "agent_faltante", "has_agent"],
    "days_in_waiting_list": ["days_in_waiting_list", "was_in_waiting_list"],
    "customer_type": ["customer_type"],
    "adr": ["adr", "adr_per_guest", "total_revenue", "flag_adr_no_positivo"],
    "total_of_special_requests": ["total_of_special_requests", "has_special_requests"],
    "company_registrado": ["company_registrado"],
}

NIVELES_RIESGO = [  # (límite superior, nivel, color, acción sugerida)
    (0.30, "Bajo", "#008009", "Gestión estándar: confirmación automática de la reserva."),
    (0.60, "Medio", "#FEBB02", "Enviar recordatorio previo a la llegada y ofrecer servicios "
                               "adicionales para aumentar el compromiso."),
    (1.01, "Alto", "#D4111E", "Solicitar prepago o depósito, contactar al cliente y considerar la "
                              "habitación en la estrategia de sobreventa (overbooking)."),
]

ACCIONES_SEGMENTO = {
    "Planificadores": "Segmento prioritario: políticas de depósito, prepago parcial y recordatorios "
                      "antes de la llegada.",
    "negocios": "Tarifas de último minuto, check-in ágil y convenios corporativos.",
    "premium": "Mayor valor por noche: servicios adicionales, upgrades y venta cruzada.",
    "fin de semana": "Paquetes de fin de semana y promociones de escapada.",
    "cortesía": "No genera tarifa: tratar por separado en los análisis de ingresos.",
}


# --------------------------------------------------------------------------
# Lectura de artefactos
# --------------------------------------------------------------------------
def existe(rel: str) -> bool:
    return (ARTEFACTOS / rel).exists()


@lru_cache(maxsize=None)
def json_(rel: str) -> dict:
    with open(ARTEFACTOS / rel, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def csv(rel: str) -> pd.DataFrame:
    return pd.read_csv(ARTEFACTOS / rel)


@lru_cache(maxsize=None)
def modelo(rel: str):
    return joblib.load(ARTEFACTOS / rel)


def artefactos_disponibles() -> dict:
    return {t: existe(f"{t}/metadata.json") for t in ("cancelacion", "lead_time", "segmentacion")} | \
           {"eda": existe("eda/kpis.json")}


def campos_usados(tarea: str) -> set[str]:
    """Campos del formulario que alimentan alguna variable del modelo campeón."""
    if not existe(f"{tarea}/metadata.json"):
        return set()
    variables = set(json_(f"{tarea}/metadata.json")["variables"])
    return {campo for campo, deps in DEPENDENCIAS.items() if variables & set(deps)}


# --------------------------------------------------------------------------
# Construcción de la reserva desde el formulario
# --------------------------------------------------------------------------
def campos_a_reserva(valores: dict) -> dict:
    """Traduce los valores del formulario al formato de columnas del dataset."""
    v = dict(valores)
    fecha = pd.Timestamp(v.pop("fecha_llegada", None) or "2016-08-15")
    v["arrival_date_year"] = fecha.year
    v["arrival_date_month"] = MESES[fecha.month - 1]
    v["arrival_date_day_of_month"] = fecha.day
    v["is_repeated_guest"] = int(bool(v.get("is_repeated_guest")))
    v["company_registrado"] = int(bool(v.get("company_registrado")))
    v["agent"] = int(v.get("agent") or 0)
    v["assigned_room_type"] = v.get("reserved_room_type")
    for c in list(v):
        if v[c] is None or v[c] == "":
            v.pop(c)
    return v


def _fila_modelo(tarea: str, valores: dict) -> pd.DataFrame:
    meta = json_(f"{tarea}/metadata.json")
    fila = construir_reserva(campos_a_reserva(valores))
    fila = fila.copy()
    if "agent" in fila.columns:
        fila["agent"] = fila["agent"].fillna(0).astype(int).astype(str)
    X = fila[meta["variables"]].copy()
    for c, spec in meta["especificacion"].items():
        if spec["tipo"] == "categorica":
            X[c] = X[c].astype(str)
    return X


# --------------------------------------------------------------------------
# Predicciones
# --------------------------------------------------------------------------
def predecir_cancelacion(valores: dict) -> dict:
    X = _fila_modelo("cancelacion", valores)
    proba = float(modelo("cancelacion/campeon.joblib").predict_proba(X)[0, 1])
    for limite, nivel, color, accion in NIVELES_RIESGO:
        if proba < limite:
            break
    meta = json_("cancelacion/metadata.json")
    return {"proba": proba, "nivel": nivel, "color": color, "accion": accion,
            "tasa_base": meta["metricas"]["tasa_cancelacion_test"],
            "umbral": meta.get("umbral_decision", 0.5)}


def predecir_lead_time(valores: dict) -> dict:
    X = _fila_modelo("lead_time", valores)
    dias = float(np.clip(modelo("lead_time/campeon.joblib").predict(X)[0], 0, None))
    meta = json_("lead_time/metadata.json")
    fecha = pd.Timestamp(valores.get("fecha_llegada") or "2016-08-15")
    mae = meta["metricas"]["MAE_test"]
    return {"dias": dias, "mae": mae, "fecha_reserva": fecha - pd.Timedelta(days=round(dias)),
            "rango": (max(dias - mae, 0), dias + mae)}


def metricas_umbral(umbral: float) -> dict:
    """Recalcula la matriz de confusión del campeón para cualquier umbral (datos de test)."""
    df = csv("cancelacion/probabilidades_test.csv")
    pred = (df["proba"] >= umbral).astype(int)
    real = df["real"]
    vp = int(((pred == 1) & (real == 1)).sum())
    fp = int(((pred == 1) & (real == 0)).sum())
    fn = int(((pred == 0) & (real == 1)).sum())
    vn = int(((pred == 0) & (real == 0)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    return {"VP": vp, "FP": fp, "FN": fn, "VN": vn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "alertas": vp + fp, "total": len(df)}


def asignar_segmento(lead_time, noches_fin_semana, noches_semana, adr, huespedes, solicitudes) -> dict:
    asignador = modelo("segmentacion/asignador.joblib")
    seg = asignador.predecir_reserva(lead_time, noches_fin_semana, noches_semana, adr, huespedes,
                                     solicitudes)
    nombre = asignador.nombres.get(seg, f"Segmento S{seg}")
    perfil = csv("segmentacion/perfil_segmentos.csv").set_index("segmento")
    accion = next((a for clave, a in ACCIONES_SEGMENTO.items() if clave.lower() in nombre.lower()),
                  "Revisar el perfil del segmento para definir la acción comercial.")
    return {"segmento": seg, "nombre": nombre, "accion": accion,
            "perfil": perfil.loc[seg].to_dict() if seg in perfil.index else {}}


def verificar_versiones() -> list[str]:
    """Compara las versiones instaladas con las usadas al entrenar cada modelo."""
    from importlib import metadata as md
    avisos = []
    for tarea in ("cancelacion", "lead_time", "segmentacion"):
        if not existe(f"{tarea}/metadata.json"):
            continue
        for paquete, version in json_(f"{tarea}/metadata.json").get("versiones", {}).items():
            if paquete in ("python", "shap") or version is None:
                continue
            try:
                instalada = md.version(paquete)
            except md.PackageNotFoundError:
                instalada = None
            if instalada != version:
                avisos.append(f"{tarea}: {paquete} entrenado con {version}, instalado {instalada}")
    return avisos
