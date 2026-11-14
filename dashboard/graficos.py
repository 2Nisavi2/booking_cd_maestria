"""Constructores de gráficos Plotly del tablero."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard import tema as T

ETIQUETAS_VARIABLES = {
    "country": "País de origen", "agent": "Agencia", "lead_time": "Antelación",
    "total_of_special_requests": "Solicitudes especiales", "market_segment": "Segmento de mercado",
    "arrival_date_year": "Año de llegada", "arrival_date_month": "Mes de llegada",
    "customer_type": "Tipo de cliente", "booking_changes": "Cambios en la reserva", "adr": "Tarifa (ADR)",
    "adr_per_guest": "Tarifa por huésped", "previous_cancel_rate": "Tasa de cancelación previa",
    "deposit_type": "Tipo de depósito", "distribution_channel": "Canal de distribución",
    "stays_in_week_nights": "Noches entre semana", "stays_in_weekend_nights": "Noches de fin de semana",
    "arrival_date_day_of_month": "Día del mes", "arrival_weekday": "Día de la semana",
    "weekend_ratio": "Proporción fin de semana", "hotel": "Hotel", "meal": "Régimen de comidas",
    "reserved_room_type": "Tipo de habitación", "adults": "Adultos", "children": "Niños",
    "babies": "Bebés", "is_repeated_guest": "Huésped repetido",
    "previous_cancellations": "Cancelaciones previas",
    "previous_bookings_not_canceled": "Reservas previas no canceladas",
    "days_in_waiting_list": "Días en lista de espera", "company_registrado": "Reserva de empresa",
    "has_kids": "Viaja con niños", "arrival_is_weekend": "Llega en fin de semana",
    "flag_adr_no_positivo": "Tarifa no positiva", "is_canceled": "Cancelación",
    "required_car_parking_spaces": "Parqueadero", "reservation_status": "Estado de la reserva",
    "reservation_status_date": "Fecha de estado", "assigned_room_type": "Habitación asignada",
    "room_changed": "Cambio de habitación", "needs_parking": "Requiere parqueadero",
}


def etiqueta(v: str) -> str:
    return ETIQUETAS_VARIABLES.get(v, v)


def pareto_importancia(tabla: pd.DataFrame, metodo: str, umbral: float = 0.90) -> go.Figure:
    """Barras de importancia (SHAP o permutación) con la curva acumulada y el corte de selección."""
    t = tabla.copy()
    t["nombre"] = t["variable"].map(etiqueta)
    colores = [T.AZUL if s else T.GRIS_CLARO for s in t["seleccionada"]]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=t["nombre"], y=t["importancia_%"], marker_color=colores, name="Importancia",
                hovertemplate="<b>%{x}</b><br>Importancia: %{y:.1f} %<extra></extra>")
    fig.add_scatter(x=t["nombre"], y=t["acumulado_%"], mode="lines+markers", name="Acumulado",
                    line=dict(color=T.AMARILLO, width=3), marker=dict(size=6, color=T.AZUL_OSCURO),
                    hovertemplate="Acumulado: %{y:.1f} %<extra></extra>", secondary_y=True)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, yref="y2", y0=umbral * 100, y1=umbral * 100,
                  line=dict(color=T.AZUL_OSCURO, dash="dash", width=1.5))
    fig.add_annotation(xref="paper", x=1, yref="y2", y=umbral * 100, text=f"Corte {umbral:.0%}",
                       showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=T.AZUL_OSCURO))
    n = int(t["seleccionada"].sum())
    fig.update_layout(title=f"{n} de {len(t)} variables concentran el {umbral:.0%} de la importancia "
                            f"({metodo})", height=430, showlegend=False, bargap=0.25)
    fig.update_yaxes(title_text="Importancia (%)", secondary_y=False)
    fig.update_yaxes(title_text="Acumulado (%)", range=[0, 105], secondary_y=True, showgrid=False)
    fig.update_xaxes(tickangle=-40)
    return fig


def beeswarm_shap(muestra: pd.DataFrame, variables: list[str]) -> go.Figure:
    """Dispersión de valores SHAP por variable coloreada por el valor (bajo = azul, alto = amarillo)."""
    fig = go.Figure()
    for i, v in enumerate(reversed(variables)):
        d = muestra[muestra["variable"] == v].sample(min(400, (muestra["variable"] == v).sum()),
                                                      random_state=0)
        jitter = (pd.Series(range(len(d))).sample(frac=1, random_state=i).values / max(len(d), 1) - 0.5) * 0.6
        color = d["valor_norm"].fillna(0.5)
        fig.add_scatter(x=d["shap"], y=[i + j for j in jitter], mode="markers", showlegend=False,
                        marker=dict(size=5, color=color, colorscale=[[0, T.AZUL], [0.5, "#C9C9C9"], [1, T.AMARILLO]],
                                    opacity=0.75, cmin=0, cmax=1, showscale=(i == 0),
                                    colorbar=dict(title="Valor", tickvals=[0, 1], ticktext=["Bajo", "Alto"],
                                                  len=0.6, thickness=10)),
                        text=d["valor"], hovertemplate=f"<b>{etiqueta(v)}</b><br>Valor: %{{text}}<br>"
                                                        "Efecto SHAP: %{x:.3f}<extra></extra>")
    fig.add_vline(x=0, line_color=T.GRIS)
    fig.update_yaxes(tickvals=list(range(len(variables))), ticktext=[etiqueta(v) for v in reversed(variables)],
                     showgrid=False)
    fig.update_layout(title="Dirección del efecto: cada punto es una reserva de la muestra de entrenamiento",
                      height=60 + 34 * len(variables), xaxis_title="Efecto sobre la predicción (valor SHAP)")
    return fig


def barras_h(etiquetas, valores, colores=None, titulo=None, sufijo="", alto=380, texto=None):
    fig = go.Figure(go.Bar(y=etiquetas, x=valores, orientation="h",
                           marker_color=colores or T.AZUL,
                           text=texto if texto is not None else [f"{v:.1f}{sufijo}".replace(".", ",") for v in valores],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:.1f}" + sufijo + "<extra></extra>"))
    fig.update_layout(title=titulo, height=alto, yaxis=dict(autorange="reversed"),
                      xaxis=dict(showgrid=True, gridcolor="#EEF0F3"))
    return fig


def gauge_probabilidad(proba: float, color: str, tasa_base: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=proba * 100,
        number=dict(suffix=" %", valueformat=".1f", font=dict(size=44, color=T.AZUL_OSCURO)),
        gauge=dict(axis=dict(range=[0, 100], ticksuffix=" %"), bar=dict(color=color, thickness=0.35),
                   steps=[dict(range=[0, 30], color="#E6F4E6"), dict(range=[30, 60], color="#FFF5D6"),
                          dict(range=[60, 100], color="#FCE4E5")],
                   threshold=dict(line=dict(color=T.AZUL_OSCURO, width=3), thickness=0.8,
                                  value=tasa_base * 100))))
    fig.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=0))
    return fig
