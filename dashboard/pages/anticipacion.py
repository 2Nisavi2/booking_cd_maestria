"""Fases 4-5 · Modelado y evaluación: anticipación de la reserva (lead_time) + simulador."""
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from dashboard import formulario as F
from dashboard import graficos as G
from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import encabezado_pagina, grafico, kpi, nota, num, sin_artefactos, tabla_simple, tarjeta

RUTA = "/anticipacion"
NOMBRE = "Predicción de anticipación"
PREFIJO = "ant"
# Variables que no se conocen al momento de estimar la antelación (o son el objetivo)
EXCLUIR = ("lead_time", "booking_changes", "days_in_waiting_list")


def _comparacion_mae(comp):
    d = comp.sort_values("MAE_test", ascending=False)
    etiquetas = [f"{m} · {e}" for m, e in zip(d["modelo"], d["etapa"])]
    colores = [T.AZUL if c else T.GRIS_CLARO for c in d["campeon"]]
    fig = G.barras_h(etiquetas, d["MAE_test"], colores, sufijo=" días", alto=460,
                     titulo="Error medio absoluto en test (menor es mejor; azul = campeón)")
    fig.update_layout(yaxis=dict(autorange=True))
    return fig


def _real_vs_predicho():
    p = S.csv("lead_time/predicciones_test.csv")
    lim = float(min(p[["real", "predicho"]].max().max(), 600))
    fig = go.Figure()
    fig.add_scatter(x=p["real"], y=p["predicho"], mode="markers",
                    marker=dict(size=5, color=T.AZUL, opacity=0.35), name="Reserva de prueba",
                    hovertemplate="Real: %{x:.0f} días<br>Predicho: %{y:.0f} días<extra></extra>")
    fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=T.AMARILLO, width=3, dash="dash"),
                    name="Predicción perfecta")
    fig.update_layout(title="Real vs. predicho: el error crece en las reservas muy anticipadas", height=460,
                      xaxis=dict(title="Antelación real (días)", range=[0, lim]),
                      yaxis=dict(title="Antelación predicha (días)", range=[0, lim]))
    return fig


def layout():
    if not S.existe("lead_time/metadata.json"):
        return html.Div([encabezado_pagina("4-5 · Modelado y evaluación", NOMBRE, ""), sin_artefactos("lead_time")])
    meta = S.json_("lead_time/metadata.json")
    m = meta["metricas"]
    comp = S.csv("lead_time/comparacion_modelos.csv")
    comp = comp[comp["modelo"] != "Baseline (media)"]
    imp = S.csv("lead_time/importancia_shap.csv")
    graf_shap = [grafico(G.pareto_importancia(imp, meta["importancia"]["metodo"].split(" (")[0]))]
    if S.existe("lead_time/shap_muestra.csv"):
        sel = imp.loc[imp["seleccionada"], "variable"].head(12).tolist()
        graf_shap.append(grafico(G.beeswarm_shap(S.csv("lead_time/shap_muestra.csv"), sel)))
    mejora = 100 * (1 - m["MAE_test"] / m["MAE_baseline"])
    return html.Div([
        encabezado_pagina(
            "4-5 · Modelado y evaluación", "¿Con cuánta antelación llegará la demanda?",
            f"El campeón ({meta['campeon']['modelo']}) estima la antelación con un error medio de "
            f"{num(m['MAE_test'])} días, {num(mejora)} % menos que una estimación ingenua. La antelación depende "
            "sobre todo de la temporada, la tarifa, la duración de la estancia y la agencia."),
        html.Div(className="fila-kpi", children=[
            kpi(f"{num(m['MAE_test'], 1)} días", "Error medio (MAE)", f"Baseline: {num(m['MAE_baseline'], 1)} días",
                acento=True),
            kpi(f"{m['R2_test']:.2f}".replace(".", ","), "R² en test", "Proporción de la variación explicada"),
            kpi(f"{num(m['RMSE_test'], 1)} días", "RMSE", "Penaliza los errores grandes"),
            kpi(str(meta["campeon"]["n_variables"]), "Variables del campeón", meta["campeon"]["etapa"]),
        ]),
        nota(meta["justificacion"], "info"),
        html.H2("Simulador: antelación esperada de una reserva", className="titulo-seccion"),
        html.Div(className="rejilla-simulador", children=[
            tarjeta("Datos de la reserva", F.construir(PREFIJO, "lead_time", EXCLUIR),
                    html.Button("Estimar antelación", id=f"{PREFIJO}-boton", className="boton")),
            tarjeta("Resultado", dcc.Loading(html.Div(id=f"{PREFIJO}-resultado",
                                                      children=nota("Completa los datos y presiona «Estimar».")),
                                             color=T.AZUL), clase="tarjeta-resultado"),
        ]),
        html.H2("¿Por qué ganó este modelo?", className="titulo-seccion"),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_comparacion_mae(comp))),
            tarjeta(None, grafico(_real_vs_predicho())),
        ]),
        tarjeta("Importancia de las variables", *graf_shap),
        tarjeta("Detalle de la comparación", tabla_simple(
            comp, {"etapa": "Etapa", "modelo": "Modelo", "n_variables": "Variables", "R2_test": "R²",
                   "MAE_test": "MAE (días)", "RMSE_test": "RMSE", "brecha_R2": "Brecha R²"},
            {c: (lambda x: f"{x:.3f}".replace(".", ",")) for c in ["R2_test", "MAE_test", "RMSE_test", "brecha_R2"]}),
            nota("El modelo con variables SHAP solo gana si su error está dentro del 2 % del mejor. La partición "
                 "fue aleatoria: por el peso del año de llegada, se recomienda validar con un corte temporal.",
                 "alerta")),
    ])


@callback(Output(f"{PREFIJO}-resultado", "children"), Input(f"{PREFIJO}-boton", "n_clicks"),
          *F.estados(PREFIJO, EXCLUIR), prevent_initial_call=True)
def predecir(_, *args):
    valores = F.valores(EXCLUIR, args)
    try:
        r = S.predecir_lead_time(valores)
    except Exception as e:  # noqa: BLE001
        return nota(f"No fue posible calcular la predicción: {e}", "alerta")
    bajo, alto = r["rango"]
    return html.Div([
        html.Div(className="resultado-grande", children=[
            html.Div(f"{num(r['dias'])} días", className="resultado-valor"),
            html.Div("de antelación esperada", className="resultado-etiqueta"),
        ]),
        html.P(["La reserva se haría alrededor del ", html.B(r["fecha_reserva"].strftime("%d/%m/%Y")),
                f" (rango típico: {num(bajo)} a {num(alto)} días, ± error medio del modelo)."]),
        nota("Uso sugerido: proyectar la curva de llenado de la fecha de llegada y planear tarifas y "
             "personal según cuándo se espera que entren las reservas.", "ok"),
    ])
