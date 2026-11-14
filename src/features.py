"""Variables derivadas y bases de modelado.

Este módulo es compartido por el entrenamiento y por el tablero: así las
variables que ve el modelo en producción se calculan EXACTAMENTE igual que en
el entrenamiento.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MESES = ["January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"]
MESES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
            "Septiembre", "Octubre", "Noviembre", "Diciembre"]
NUM_MES = {m: i for i, m in enumerate(MESES, 1)}


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Añade 31 variables de negocio a partir de las columnas originales (imputadas)."""
    d = df.copy()
    # --- Fechas
    d["arrival_month_num"] = d["arrival_date_month"].map(NUM_MES)
    d["arrival_date"] = pd.to_datetime(dict(year=d["arrival_date_year"], month=d["arrival_month_num"],
                                            day=d["arrival_date_day_of_month"]), errors="coerce")
    d["booking_date"] = d["arrival_date"] - pd.to_timedelta(d["lead_time"], unit="D")
    d["arrival_weekday"] = d["arrival_date"].dt.dayofweek  # 0 = lunes
    d["arrival_is_weekend"] = d["arrival_weekday"].isin([4, 5]).astype(int)
    d["arrival_quarter"] = d["arrival_date"].dt.quarter
    d["season"] = d["arrival_month_num"].map({12: "Invierno", 1: "Invierno", 2: "Invierno",
                                              3: "Primavera", 4: "Primavera", 5: "Primavera",
                                              6: "Verano", 7: "Verano", 8: "Verano",
                                              9: "Otoño", 10: "Otoño", 11: "Otoño"})
    d["high_season"] = d["arrival_month_num"].isin([7, 8]).astype(int)
    # --- Estancia
    d["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    d["weekend_ratio"] = np.where(d["total_nights"] > 0,
                                  d["stays_in_weekend_nights"] / d["total_nights"].replace(0, 1), 0)
    d["stay_type"] = pd.cut(d["total_nights"], [-1, 0, 3, 7, np.inf],
                            labels=["Sin noches", "Corta (1-3)", "Media (4-7)", "Larga (8+)"])
    # --- Huéspedes
    d["total_guests"] = d["adults"] + d["children"].fillna(0) + d["babies"]
    d["has_kids"] = ((d["children"].fillna(0) + d["babies"]) > 0).astype(int)
    d["guest_type"] = np.select([d["total_guests"] == 0, d["has_kids"] == 1,
                                 d["adults"] == 1, d["adults"] == 2],
                                ["Sin huéspedes", "Familia", "Individual", "Pareja"], "Grupo")
    # --- Precio
    d["total_revenue"] = d["adr"] * d["total_nights"]
    d["adr_per_guest"] = np.where(d["total_guests"] > 0,
                                  d["adr"] / d["total_guests"].replace(0, 1), np.nan)
    # --- Historial del cliente
    d["total_previous_bookings"] = d["previous_cancellations"] + d["previous_bookings_not_canceled"]
    d["previous_cancel_rate"] = np.where(d["total_previous_bookings"] > 0,
                                         d["previous_cancellations"]
                                         / d["total_previous_bookings"].replace(0, 1), 0)
    d["has_previous_cancel"] = (d["previous_cancellations"] > 0).astype(int)
    # --- Reserva
    d["lead_time_cat"] = pd.cut(d["lead_time"], [-1, 7, 30, 90, 180, np.inf],
                                labels=["<1 sem", "≤1 mes", "≤3 meses", "≤6 meses", ">6 meses"])
    d["has_special_requests"] = (d["total_of_special_requests"] > 0).astype(int)
    d["needs_parking"] = (d["required_car_parking_spaces"] > 0).astype(int)
    d["was_in_waiting_list"] = (d["days_in_waiting_list"] > 0).astype(int)
    d["has_booking_changes"] = (d["booking_changes"] > 0).astype(int)
    d["has_agent"] = d["agent"].notna().astype(int)
    d["is_portugal"] = (d["country"] == "PRT").astype(int)
    d["has_deposit"] = (d["deposit_type"] != "No Deposit").astype(int)
    # --- Posible fuga (ocurre al llegar): no usar para predecir cancelación
    d["room_changed"] = (d["reserved_room_type"] != d["assigned_room_type"]).astype(int)
    # --- Calidad del dato
    d["flag_sin_huespedes"] = (d["total_guests"] == 0).astype(int)
    d["flag_adr_no_positivo"] = (d["adr"] < 0).astype(int)
    d["flag_sin_noches"] = (d["total_nights"] == 0).astype(int)
    return d


# --------------------------------------------------------------------------
# Bases de modelado (exclusiones por fuga, fechas y redundancia)
# --------------------------------------------------------------------------
FUGA_LEAD_TIME = ["lead_time_cat", "booking_date", "arrival_date", "reservation_status",
                  "reservation_status_date", "assigned_room_type", "room_changed", "is_canceled",
                  "required_car_parking_spaces", "needs_parking", "booking_changes",
                  "has_booking_changes", "days_in_waiting_list", "was_in_waiting_list"]
REDUNDANTES_LEAD_TIME = ["has_deposit", "has_previous_cancel", "has_agent", "agent_faltante",
                         "total_nights", "is_portugal", "has_special_requests", "total_guests",
                         "total_previous_bookings", "stay_type", "guest_type", "arrival_month_num",
                         "arrival_quarter", "season", "high_season", "total_revenue",
                         "flag_sin_huespedes", "flag_sin_noches", "arrival_date_week_number"]

FUGA_CANCELACION = ["reservation_status", "reservation_status_date", "assigned_room_type",
                    "room_changed", "required_car_parking_spaces", "needs_parking"]
FECHAS = ["arrival_date", "booking_date"]
REDUNDANTES_CANCELACION = ["lead_time_cat", "has_deposit", "is_portugal", "has_special_requests",
                           "has_booking_changes", "has_previous_cancel", "has_agent",
                           "agent_faltante", "total_nights", "total_guests",
                           "total_previous_bookings", "stay_type", "guest_type",
                           "arrival_month_num", "arrival_quarter", "season", "high_season",
                           "arrival_date_week_number", "was_in_waiting_list", "total_revenue",
                           "flag_sin_huespedes", "flag_sin_noches"]


def _agente_como_texto(d: pd.DataFrame) -> pd.DataFrame:
    if "agent" in d.columns:
        d["agent"] = d["agent"].fillna(0).astype(int).astype(str)  # código de agencia
    return d


def preparar_base_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las variables con fuga respecto a lead_time, las redundantes y las constantes."""
    quitar = FUGA_LEAD_TIME + REDUNDANTES_LEAD_TIME
    return _agente_como_texto(df.drop(columns=[c for c in quitar if c in df.columns]))


def preparar_base_cancelacion(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las variables con fuga respecto a is_canceled, fechas, redundantes y constantes."""
    quitar = FUGA_CANCELACION + FECHAS + REDUNDANTES_CANCELACION
    return _agente_como_texto(df.drop(columns=[c for c in quitar if c in df.columns]))


# --------------------------------------------------------------------------
# Construcción de una reserva a partir del formulario del tablero
# --------------------------------------------------------------------------
# Valores por defecto: la "reserva típica" del análisis descriptivo
RESERVA_TIPICA = {
    "hotel": "City Hotel", "lead_time": 69, "arrival_date_year": 2016,
    "arrival_date_month": "August", "arrival_date_week_number": 32,
    "arrival_date_day_of_month": 15, "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
    "adults": 2, "children": 0, "babies": 0, "meal": "BB", "country": "PRT",
    "market_segment": "Online TA", "distribution_channel": "TA/TO", "is_repeated_guest": 0,
    "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0,
    "deposit_type": "No Deposit", "agent": 9, "days_in_waiting_list": 0,
    "customer_type": "Transient", "adr": 95.0, "required_car_parking_spaces": 0,
    "total_of_special_requests": 0, "company_registrado": 0,
}


def construir_reserva(campos: dict) -> pd.DataFrame:
    """Convierte los campos del formulario en una fila con las mismas columnas que df_new.

    Replica la imputación aplicada en el entrenamiento:
    - agent: 0 = sin agencia (+ indicador agent_faltante)
    - company: solo el indicador company_registrado
    """
    r = {**RESERVA_TIPICA, **{k: v for k, v in campos.items() if v is not None}}
    fecha = pd.Timestamp(year=int(r["arrival_date_year"]), month=NUM_MES[r["arrival_date_month"]],
                         day=int(r["arrival_date_day_of_month"]))
    r["arrival_date_week_number"] = int(fecha.isocalendar().week)
    agente = int(r.get("agent") or 0)
    r["agent"] = agente
    r["agent_faltante"] = int(agente == 0)
    r.setdefault("assigned_room_type", r["reserved_room_type"])
    r["is_canceled"] = 0
    r["reservation_status"] = "Check-Out"
    r["reservation_status_date"] = str(fecha.date())
    fila = pd.DataFrame([r])
    for c in ["lead_time", "arrival_date_year", "arrival_date_day_of_month", "stays_in_weekend_nights",
              "stays_in_week_nights", "adults", "babies", "is_repeated_guest",
              "previous_cancellations", "previous_bookings_not_canceled", "booking_changes",
              "days_in_waiting_list", "required_car_parking_spaces", "total_of_special_requests",
              "company_registrado", "agent_faltante"]:
        fila[c] = fila[c].astype(int)
    fila["children"] = fila["children"].astype(float)
    fila["agent"] = fila["agent"].astype(float)
    fila["adr"] = fila["adr"].astype(float)
    return crear_variables_derivadas(fila)
