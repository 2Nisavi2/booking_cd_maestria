"""Formulario de reserva reutilizable para los simuladores de predicción."""
from __future__ import annotations

from datetime import date

from dash import State, dcc, html

from dashboard import servicios

PAISES = {"PRT": "Portugal", "GBR": "Reino Unido", "FRA": "Francia", "ESP": "España", "DEU": "Alemania",
          "ITA": "Italia", "IRL": "Irlanda", "BEL": "Bélgica", "BRA": "Brasil", "NLD": "Países Bajos",
          "USA": "Estados Unidos", "CHE": "Suiza", "CN": "China", "AUT": "Austria", "SWE": "Suecia",
          "CHN": "China", "POL": "Polonia", "ISR": "Israel", "RUS": "Rusia", "NOR": "Noruega",
          "ROU": "Rumania", "FIN": "Finlandia", "DNK": "Dinamarca", "AUS": "Australia", "AGO": "Angola",
          "LUX": "Luxemburgo", "MAR": "Marruecos", "TUR": "Turquía", "HUN": "Hungría", "ARG": "Argentina",
          "JPN": "Japón", "CZE": "Chequia", "IND": "India", "KOR": "Corea del Sur", "GRC": "Grecia",
          "DZA": "Argelia", "SRB": "Serbia", "HRV": "Croacia", "MEX": "México", "IRN": "Irán"}
COMIDAS = {"BB": "Alojamiento y desayuno (BB)", "HB": "Media pensión (HB)", "FB": "Pensión completa (FB)",
           "SC": "Sin comidas (SC)"}

# (campo, etiqueta, tipo, grupo, parámetros)
CAMPOS = [
    ("hotel", "Hotel", "select", "Reserva", {}),
    ("fecha_llegada", "Fecha de llegada", "fecha", "Reserva", {"valor": "2016-08-15"}),
    ("lead_time", "Antelación (días entre reserva y llegada)", "num", "Reserva",
     {"valor": 69, "min": 0, "max": 737}),
    ("stays_in_weekend_nights", "Noches de fin de semana", "num", "Reserva", {"valor": 1, "min": 0, "max": 19}),
    ("stays_in_week_nights", "Noches entre semana", "num", "Reserva", {"valor": 2, "min": 0, "max": 50}),
    ("adults", "Adultos", "num", "Huéspedes", {"valor": 2, "min": 1, "max": 10}),
    ("children", "Niños", "num", "Huéspedes", {"valor": 0, "min": 0, "max": 5}),
    ("babies", "Bebés", "num", "Huéspedes", {"valor": 0, "min": 0, "max": 3}),
    ("country", "País de origen", "select", "Huéspedes", {}),
    ("is_repeated_guest", "Huésped repetido", "check", "Huéspedes", {}),
    ("market_segment", "Segmento de mercado", "select", "Canal y condiciones", {}),
    ("distribution_channel", "Canal de distribución", "select", "Canal y condiciones", {}),
    ("agent", "Agencia (código)", "select", "Canal y condiciones", {}),
    ("company_registrado", "Reserva a nombre de empresa", "check", "Canal y condiciones", {}),
    ("customer_type", "Tipo de cliente", "select", "Canal y condiciones", {}),
    ("deposit_type", "Tipo de depósito", "select", "Canal y condiciones", {}),
    ("meal", "Régimen de comidas", "select", "Estancia", {}),
    ("reserved_room_type", "Tipo de habitación", "select", "Estancia", {}),
    ("adr", "Tarifa diaria promedio (€)", "num", "Estancia", {"valor": 95, "min": 0, "max": 500, "paso": 1}),
    ("total_of_special_requests", "Solicitudes especiales", "num", "Estancia", {"valor": 0, "min": 0, "max": 5}),
    ("booking_changes", "Cambios en la reserva", "num", "Historial", {"valor": 0, "min": 0, "max": 10}),
    ("days_in_waiting_list", "Días en lista de espera", "num", "Historial", {"valor": 0, "min": 0, "max": 400}),
    ("previous_cancellations", "Cancelaciones previas del cliente", "num", "Historial",
     {"valor": 0, "min": 0, "max": 30}),
    ("previous_bookings_not_canceled", "Reservas previas no canceladas", "num", "Historial",
     {"valor": 0, "min": 0, "max": 50}),
]
DEFECTOS = {"hotel": "City Hotel", "country": "PRT", "market_segment": "Online TA",
            "distribution_channel": "TA/TO", "agent": 9, "customer_type": "Transient",
            "deposit_type": "No Deposit", "meal": "BB", "reserved_room_type": "A"}


def _opciones(campo: str) -> list[dict]:
    op = servicios.json_("eda/opciones_formulario.json") if servicios.existe("eda/opciones_formulario.json") else {}
    if campo == "country":
        return [{"label": f"{PAISES.get(c, c)} ({c})", "value": c} for c in op.get("country", ["PRT"])]
    if campo == "agent":
        return [{"label": "Sin agencia", "value": 0}] + \
               [{"label": f"Agencia {a}", "value": a} for a in op.get("agent", [9])]
    if campo == "meal":
        return [{"label": COMIDAS.get(m, m), "value": m} for m in op.get("meal", ["BB"])]
    return [{"label": x, "value": x} for x in op.get(campo, [DEFECTOS.get(campo)])]


def _control(prefijo: str, campo: str, tipo: str, p: dict):
    id_ = f"{prefijo}-{campo}"
    if tipo == "select":
        return dcc.Dropdown(id=id_, options=_opciones(campo), value=DEFECTOS.get(campo), clearable=False)
    if tipo == "fecha":
        return dcc.DatePickerSingle(id=id_, date=p["valor"], display_format="DD/MM/YYYY",
                                    first_day_of_week=1, min_date_allowed=date(2015, 1, 1),
                                    max_date_allowed=date(2030, 12, 31))
    if tipo == "check":
        return dcc.Checklist(id=id_, options=[{"label": " Sí", "value": 1}], value=[],
                             className="check")
    return dcc.Input(id=id_, type="number", value=p["valor"], min=p.get("min"), max=p.get("max"),
                     step=p.get("paso", 1), className="input-num", debounce=True)


def construir(prefijo: str, tarea: str, excluir: tuple = ()):
    """Formulario agrupado. Marca con ★ los campos que usa el modelo campeón."""
    usados = servicios.campos_usados(tarea)
    grupos: dict[str, list] = {}
    for campo, etiqueta, tipo, grupo, p in CAMPOS:
        if campo in excluir:
            continue
        marca = html.Span(" ★", className="estrella", title="Usado por el modelo campeón") \
            if campo in usados else None
        grupos.setdefault(grupo, []).append(html.Div(className="campo", children=[
            html.Label([etiqueta, marca], htmlFor=f"{prefijo}-{campo}"), _control(prefijo, campo, tipo, p)]))
    return html.Div(className="formulario", children=[
        html.P(["Los campos con ", html.Span("★", className="estrella"),
                " alimentan las variables del modelo campeón; el resto se conserva para describir la reserva."],
               className="ayuda"),
        *[html.Fieldset([html.Legend(g), html.Div(c, className="rejilla-campos")]) for g, c in grupos.items()],
    ])


def estados(prefijo: str, excluir: tuple = ()) -> list[State]:
    estados_ = []
    for campo, _, tipo, _, _ in CAMPOS:
        if campo in excluir:
            continue
        prop = "date" if tipo == "fecha" else "value"
        estados_.append(State(f"{prefijo}-{campo}", prop))
    return estados_


def valores(excluir: tuple, args) -> dict:
    campos = [c for c, *_ in CAMPOS if c not in excluir]
    tipos = {c: t for c, _, t, _, _ in CAMPOS}
    v = {}
    for campo, val in zip(campos, args):
        if tipos[campo] == "check":
            val = 1 if val else 0
        v[campo] = val
    return v
