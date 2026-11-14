"""Fase 2 · Análisis descriptivo: qué explica la cancelación."""
import plotly.graph_objects as go
from dash import html

from dashboard import graficos as G
from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import encabezado_pagina, grafico, nota, sin_artefactos, tarjeta

RUTA = "/factores"
NOMBRE = "Qué explica la cancelación"


def _factores():
    f = S.csv("eda/cancelacion_factores.csv")
    k = S.json_("eda/kpis.json")
    colores = [T.ROJO if t > k["tasa_cancelacion"] else T.AZUL for t in f["tasa_cancelacion"]]
    fig = G.barras_h(f["factor"], f["tasa_cancelacion"], colores, sufijo=" %", alto=560,
                     titulo="Rojo: características que elevan el riesgo por encima del promedio")
    fig.add_vline(x=k["tasa_cancelacion"], line_dash="dash", line_color=T.AZUL_OSCURO,
                  annotation_text=f"Promedio {k['tasa_cancelacion']:.0f} %", annotation_position="top")
    fig.update_layout(xaxis=dict(range=[0, 112], title="Tasa de cancelación (%)"))
    return fig


def _asociacion():
    a = S.csv("eda/asociacion_cancelacion.csv").head(12).iloc[::-1]
    colores = [T.ROJO if v > 0 else T.AZUL for v in a["asociacion"]]
    fig = go.Figure(go.Bar(y=[G.etiqueta(v) for v in a["variable"]], x=a["asociacion"], orientation="h",
                           marker_color=colores, customdata=a["medida"],
                           text=[f"{v:+.2f}".replace(".", ",") for v in a["asociacion"]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>%{customdata}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=T.GRIS)
    fig.update_layout(title="Fuerza de asociación con la cancelación (rojo: la aumenta, azul: la reduce)",
                      height=460, xaxis_title="Asociación (V de Cramér en categóricas, Pearson en numéricas)")
    return fig


def _antelacion():
    g = S.csv("eda/grupos_lead_adr.csv")
    d = g[g["variable"].isin(["Segmento", "Depósito", "Resultado"])].dropna(subset=["lead_time_mediana"])
    d = d.sort_values(["variable", "lead_time_mediana"], ascending=[True, False])
    colores = {"Segmento": T.AZUL, "Depósito": T.AZUL_OSCURO, "Resultado": T.AMARILLO}
    fig = go.Figure()
    for var, sub in d.groupby("variable", sort=False):
        fig.add_bar(x=sub["categoria"], y=sub["lead_time_mediana"], name=var, marker_color=colores[var],
                    text=sub["lead_time_mediana"].round(0).astype(int), textposition="outside")
    fig.update_layout(title="Grupos y operadores reservan con meses de antelación; las canceladas también",
                      yaxis_title="Antelación mediana (días)", height=420, xaxis_tickangle=-30)
    return fig


def _tarifa_segmento():
    g = S.csv("eda/grupos_lead_adr.csv")
    d = g[g["variable"] == "Segmento"].sort_values("adr_mediana", ascending=False)
    fig = go.Figure(go.Bar(x=d["categoria"], y=d["adr_mediana"], marker_color=T.AZUL,
                           text=[f"{v:.0f} €" for v in d["adr_mediana"]], textposition="outside"))
    fig.update_layout(title="Las agencias online y la venta directa pagan las tarifas más altas",
                      yaxis_title="Tarifa diaria mediana (€)", height=420, xaxis_tickangle=-30)
    return fig


def _composicion():
    c = S.csv("eda/composicion.csv")
    fig = go.Figure()
    paleta = [T.AZUL_OSCURO, T.AZUL, T.AMARILLO, T.AZUL_MEDIO, T.GRIS_CLARO]
    for variable, sub in c.groupby("variable", sort=False):
        for i, (_, r) in enumerate(sub.iterrows()):
            fig.add_bar(y=[variable], x=[r["pct"]], orientation="h", name=r["categoria"], showlegend=False,
                        marker_color=paleta[i % len(paleta)], marker_line=dict(color="white", width=1),
                        text=f"{r['categoria']} {r['pct']:.0f} %" if r["pct"] >= 9 else "",
                        textposition="inside", insidetextanchor="middle",
                        textfont=dict(color="#1A1A1A" if paleta[i % len(paleta)] in (T.AMARILLO, T.GRIS_CLARO) else "white"),
                        hovertemplate=f"{r['categoria']}: {r['pct']:.1f} %<extra>{variable}</extra>")
    fig.update_layout(barmode="stack", title="Composición: predominan agencias, el City Hotel y la reserva sin depósito",
                      height=380, xaxis=dict(range=[0, 100], ticksuffix=" %"), yaxis=dict(autorange="reversed"))
    return fig


def _histograma():
    h = S.csv("eda/lead_time_hist.csv")
    fig = go.Figure(go.Bar(x=h["desde"] + 5, y=h["reservas"], marker_color=T.AZUL, width=9,
                           hovertemplate="%{x:.0f} días: %{y:,} reservas<extra></extra>"))
    fig.update_layout(title="Pico de reservas de última hora y cola larga hasta 737 días",
                      xaxis_title="Antelación (días, intervalos de 10)", yaxis_title="Reservas", height=380)
    return fig


def layout():
    if not S.existe("eda/cancelacion_factores.csv"):
        return html.Div([encabezado_pagina("2 · Análisis descriptivo", NOMBRE, ""), sin_artefactos("eda")])
    return html.Div([
        encabezado_pagina(
            "2 · Análisis descriptivo", "Compromiso frente a incertidumbre",
            "Las reservas con señales de compromiso (solicitudes especiales, cambios, cliente repetido) casi no "
            "se cancelan; las de alta incertidumbre (antelación larga, grupos, depósito no reembolsable) sí."),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_factores())),
            html.Div([tarjeta(None, grafico(_asociacion())),
                      nota("Los depósitos no reembolsables cancelan el 99 % porque corresponden a bloqueos de "
                           "grupos y operadores que el hotel ya cobró: es un predictor fuerte que debe "
                           "interpretarse con cuidado.", "alerta")]),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_antelacion())),
            tarjeta(None, grafico(_tarifa_segmento())),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_composicion())),
            tarjeta(None, grafico(_histograma())),
        ]),
    ])
