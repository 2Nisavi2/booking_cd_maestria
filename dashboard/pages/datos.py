"""Fases 2-3 · Comprensión y preparación de los datos."""
import plotly.graph_objects as go
from dash import html

from dashboard import graficos as G
from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import encabezado_pagina, grafico, kpi, nota, num, pct, sin_artefactos, tabla_simple, tarjeta

RUTA = "/datos"
NOMBRE = "Calidad de los datos"


def _faltantes():
    f = S.csv("eda/faltantes.csv").sort_values("pct_faltantes")
    estructural = {"company", "agent"}
    colores = [T.GRIS_CLARO if v in estructural else T.AZUL for v in f["variable"]]
    texto = [f"{p:.2f} %".replace(".", ",") if p >= 0.01 else f"{int(n)} registros"
             for p, n in zip(f["pct_faltantes"], f["n_faltantes"])]
    fig = go.Figure(go.Bar(y=f["variable"], x=f["pct_faltantes"], orientation="h", marker_color=colores,
                           text=texto, textposition="outside", cliponaxis=False))
    fig.update_layout(title="Solo 4 de 32 variables tienen faltantes; en gris, los que significan "
                            "«sin intermediario»", xaxis_title="% de reservas", height=300,
                      xaxis=dict(range=[0, 110]))
    return fig


def _anios():
    a = S.csv("eda/reservas_anio.csv")
    etiquetas = {2015: "2015 (jul–dic)", 2016: "2016 (completo)", 2017: "2017 (ene–ago)"}
    fig = go.Figure(go.Bar(x=[etiquetas.get(x, str(x)) for x in a["arrival_date_year"]], y=a["reservas"],
                           marker_color=[T.AZUL if x == 2016 else T.GRIS_CLARO for x in a["arrival_date_year"]],
                           text=[num(v) for v in a["reservas"]], textposition="outside"))
    fig.update_layout(title="Solo 2016 es un año completo: los volúmenes anuales no son comparables",
                      height=300, yaxis_title="Reservas")
    return fig


def _fuga():
    f = S.csv("eda/fuga.csv").sort_values("auc_univariado")
    colores = [T.ROJO if a == "Excluida" else T.AZUL for a in f["accion"]]
    fig = go.Figure(go.Bar(y=[G.etiqueta(v) for v in f["variable"]], x=f["auc_univariado"], orientation="h",
                           marker_color=colores, customdata=f["motivo"],
                           text=[f"{v:.3f}".replace(".", ",") for v in f["auc_univariado"]],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="<b>%{y}</b><br>AUC: %{x:.3f}<br>%{customdata}<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dot", line_color=T.GRIS)
    fig.update_layout(title="Rojo: variables excluidas porque se conocen después de reservar (fuga)",
                      xaxis=dict(range=[0.45, 1.08], title="AUC univariado frente a la cancelación"), height=480)
    return fig


def layout():
    if not S.existe("eda/kpis.json"):
        return html.Div([encabezado_pagina("2 · Comprensión de los datos", NOMBRE, ""), sin_artefactos("eda")])
    k = S.json_("eda/kpis.json")
    imp = S.csv("eda/imputacion.csv")
    return html.Div([
        encabezado_pagina(
            "2-3 · Comprensión y preparación de los datos", "Datos confiables antes de modelar",
            "Los faltantes tienen significado de negocio, una cuarta parte de las filas está duplicada y "
            "algunas variables «adivinan» el resultado. Todo se corrigió antes de entrenar."),
        html.Div(className="fila-kpi", children=[
            kpi(num(k["n_reservas"]), "Reservas", "32 variables originales"),
            kpi(num(k["n_duplicados"]), "Filas duplicadas", pct(k["pct_duplicados"]) + " del total"),
            kpi(pct(k["tasa_cancelacion"]), "Cancelación con duplicados", "Base original"),
            kpi(pct(k["tasa_cancelacion_sin_duplicados"]), "Cancelación sin duplicados",
                "Los duplicados inflaban el problema", acento=True),
            kpi("31", "Variables de negocio creadas", "Fechas, estancia, huéspedes, precio, historial"),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_faltantes())),
            tarjeta(None, grafico(_anios())),
        ]),
        tarjeta("Tratamiento de faltantes según su diagnóstico", tabla_simple(
            imp, {"variable": "Variable", "mecanismo": "Diagnóstico", "pct_imputado": "% faltante",
                  "metodo": "Tratamiento", "efecto_en_distribucion": "Efecto"},
            {"pct_imputado": lambda x: pct(x, 2)}),
            nota("El test de Little y modelos que predicen el faltante (AUC ≈ 0,95) confirman que los vacíos "
                 "no son aleatorios: describen el canal de venta. Se conservaron las 119.390 reservas.")),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_fuga())),
            tarjeta("¿Qué es la fuga de información?", html.P(
                "Una variable con fuga contiene información que no existe al momento de predecir. "
                "Incluirla produce un modelo «perfecto» en el papel e inútil en la operación."), html.Ul([
                html.Li([html.B("Estado de la reserva: "), "es el propio resultado (AUC = 1,0)."]),
                html.Li([html.B("Habitación asignada y cambio de habitación: "), "se conocen al llegar."]),
                html.Li([html.B("Parqueadero: "), "las reservas que lo piden nunca se cancelan; probablemente "
                                                  "se registra al llegar, por lo que se excluyó."]),
            ]), nota("Los predictores legítimos más fuertes por sí solos son el país, la antelación, el tipo de "
                     "depósito, las solicitudes especiales y el segmento de mercado.", "ok")),
        ]),
    ])
