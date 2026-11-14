"""Fase 1 · Comprensión del negocio: el problema."""
import plotly.graph_objects as go
from dash import dcc, html
from plotly.subplots import make_subplots

from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import encabezado_pagina, grafico, kpi, num, pct, sin_artefactos, tarjeta

RUTA = "/"
NOMBRE = "El problema"


def _demanda_mensual():
    m = S.csv("eda/mensual_hotel.csv")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for hotel, color in T.HOTEL.items():
        d = m[m["hotel"] == hotel]
        fig.add_bar(x=d["mes"], y=d["reservas"], name=f"Reservas · {hotel}", marker_color=color,
                    hovertemplate="%{x}: %{y:,} reservas<extra>" + hotel + "</extra>")
    for hotel, color in [("City Hotel", T.AZUL_OSCURO), ("Resort Hotel", T.ROJO)]:
        d = m[m["hotel"] == hotel]
        fig.add_scatter(x=d["mes"], y=d["tasa_cancelacion"], name=f"% cancelación · {hotel}",
                        mode="lines+markers", line=dict(color=color, width=2, dash="dot"), secondary_y=True,
                        hovertemplate="%{x}: %{y:.1f} % cancelación<extra>" + hotel + "</extra>")
    fig.update_layout(title="La demanda tiene picos en verano y la cancelación es mayor en el City Hotel",
                      barmode="group", height=400)
    fig.update_yaxes(title_text="Reservas", secondary_y=False)
    fig.update_yaxes(title_text="% cancelación", range=[0, 60], secondary_y=True, showgrid=False)
    return fig


def _tarifa_mensual():
    m = S.csv("eda/mensual_hotel.csv")
    fig = go.Figure()
    for hotel, color in [("City Hotel", T.AZUL), ("Resort Hotel", T.AMARILLO)]:
        d = m[m["hotel"] == hotel]
        fig.add_scatter(x=d["mes"], y=d["adr_mediana"], name=hotel, mode="lines+markers",
                        line=dict(color=color, width=3), marker=dict(size=8),
                        hovertemplate="%{x}: %{y:.0f} €<extra>" + hotel + "</extra>")
    fig.update_layout(title="La tarifa del Resort se multiplica en verano; la del City es estable",
                      yaxis_title="Tarifa diaria mediana (€)", height=400)
    return fig


def layout():
    if not S.existe("eda/kpis.json"):
        return html.Div([encabezado_pagina("1 · Comprensión del negocio", "El problema", ""), sin_artefactos("eda")])
    k = S.json_("eda/kpis.json")
    return html.Div([
        encabezado_pagina(
            "1 · Comprensión del negocio", "Más de una de cada tres reservas se cancela",
            "Dos hoteles en Portugal (Lisboa y el Algarve) pierden ingresos por cancelaciones y dependen de "
            "intermediarios. Este tablero recorre el análisis CRISP-DM y permite simular predicciones."),
        html.Div(className="fila-kpi", children=[
            kpi(num(k["n_reservas"]), "Reservas analizadas", k["periodo"]),
            kpi(pct(k["tasa_cancelacion"]), "Tasa de cancelación",
                f"{pct(k['cancelacion_hotel']['City Hotel'])} City · {pct(k['cancelacion_hotel']['Resort Hotel'])} Resort",
                acento=True),
            kpi(pct(k["pct_agencias"]), "Reservas vía agencias/operadores", "Alta dependencia de intermediarios"),
            kpi(f"{num(k['lead_time_mediana'])} días", "Antelación mediana", f"Media de {num(k['lead_time_media'])} días"),
            kpi(f"{num(k['adr_mediana'], 1)} €", "Tarifa diaria mediana", "ADR, en euros"),
            kpi(pct(k["pct_repetidos"]), "Huéspedes repetidos", "Casi todos son clientes nuevos"),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_demanda_mensual())),
            tarjeta(None, grafico(_tarifa_mensual())),
        ]),
        html.Div(className="rejilla-3", children=[
            tarjeta("Pregunta de negocio",
                    html.P("¿Qué parámetros de la demanda histórica, las cancelaciones y los perfiles de "
                           "cliente debe tener en cuenta el hotel para decidir mejor?")),
            tarjeta("Tres objetivos, tres modelos", html.Ul([
                html.Li([html.B("Anticipación: "), "estimar con cuántos días se reserva (regresión)."]),
                html.Li([html.B("Cancelación: "), "estimar la probabilidad de cancelar (clasificación)."]),
                html.Li([html.B("Perfiles: "), "agrupar las reservas por comportamiento (segmentación)."]),
            ])),
            tarjeta("Cómo leer este tablero", html.Ol([
                html.Li(dcc.Link("Calidad de los datos", href="/datos")),
                html.Li(dcc.Link("Qué explica la cancelación", href="/factores")),
                html.Li(dcc.Link("Predicción de cancelación", href="/cancelacion")),
                html.Li(dcc.Link("Predicción de anticipación", href="/anticipacion")),
                html.Li(dcc.Link("Segmentos de clientes", href="/segmentos")),
                html.Li(dcc.Link("Recomendaciones", href="/recomendaciones")),
            ])),
        ]),
    ])
