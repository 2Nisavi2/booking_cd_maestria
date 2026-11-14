"""Lectura y validación de la fuente de datos original (hotel_bookings.csv)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_ESPERADAS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
    "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV original y valida su estructura.

    El dataset de Kaggle trae los nulos de agent/company como el texto "NULL";
    pandas los interpreta como NaN por defecto.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. Descarga hotel_bookings.csv desde "
            "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand "
            "y ubícalo en data/raw/ (o indica la ruta con --datos).")

    logger.info("Leyendo datos desde %s (%.1f MB)", ruta, ruta.stat().st_size / 1e6)
    df = pd.read_csv(ruta)
    faltan = sorted(set(COLUMNAS_ESPERADAS) - set(df.columns))
    if faltan:
        raise ValueError(f"Al archivo le faltan columnas esperadas: {faltan}")

    logger.info("Datos cargados: %s filas x %s columnas", f"{len(df):,}".replace(",", "."), df.shape[1])
    logger.info("Tasa de cancelación en la base original: %.1f %%", 100 * df["is_canceled"].mean())
    n_dup = int(df.duplicated().sum())
    logger.info("Filas duplicadas en la base original: %s (%.1f %%)",
                f"{n_dup:,}".replace(",", "."), 100 * n_dup / len(df))
    return df
