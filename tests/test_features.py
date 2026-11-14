import pandas as pd

from src.features import (RESERVA_TIPICA, construir_reserva, crear_variables_derivadas,
                          preparar_base_cancelacion, preparar_base_lead_time)


def test_construir_reserva_tiene_31_derivadas():
    fila = construir_reserva({})
    assert len(fila) == 1
    for c in ["total_nights", "weekend_ratio", "adr_per_guest", "previous_cancel_rate", "is_portugal",
              "arrival_weekday", "season", "guest_type"]:
        assert c in fila.columns
    assert fila.loc[0, "total_nights"] == RESERVA_TIPICA["stays_in_weekend_nights"] + RESERVA_TIPICA["stays_in_week_nights"]


def test_bases_excluyen_fuga():
    fila = construir_reserva({"lead_time": 10})
    canc = preparar_base_cancelacion(fila)
    lead = preparar_base_lead_time(fila)
    for c in ["reservation_status", "assigned_room_type", "required_car_parking_spaces"]:
        assert c not in canc.columns
    for c in ["is_canceled", "booking_changes", "lead_time_cat"]:
        assert c not in lead.columns
    assert not pd.api.types.is_numeric_dtype(canc["agent"])


def test_variables_derivadas_casos_borde():
    fila = construir_reserva({"stays_in_weekend_nights": 0, "stays_in_week_nights": 0, "adults": 0})
    assert fila.loc[0, "weekend_ratio"] == 0
    assert fila.loc[0, "flag_sin_noches"] == 1
    assert pd.isna(fila.loc[0, "adr_per_guest"])
    d = crear_variables_derivadas(fila)
    assert d.shape == fila.shape
