"""Fases 4-5 · Modelado y evaluación: propensión a cancelar + simulador."""
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from dashboard import formulario as F
from dashboard import graficos as G
from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import (encabezado_pagina, grafico, kpi, nota, num, pct, sin_artefactos,
                                   tabla_simple, tarjeta)

RUTA = "/cancelacion"
NOMBRE = "Predicción de cancelación"
PREFIJO = "can"
EXCLUIR = ()


def _roc():
    r = S.csv("cancelacion/curvas_roc.csv")
    campeon = S.json_("cancelacion/metadata.json")["campeon"]["modelo"]
    fig = go.Figure()
    for i, (modelo, d) in enumerate(r.groupby("modelo")):
        es = modelo == campeon
        fig.add_scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                        name=f"{modelo} (AUC {d['auc'].iloc[0]:.3f})".replace(".", ","),
                        line=dict(width=4 if es else 1.5, color=T.AZUL if es else T.SECUENCIA[(i + 2) % 7]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color=T.GRIS_CLARO), showlegend=False)
    fig.update_layout(title="Curvas ROC: el campeón ordena mejor el riesgo", height=420,
                      xaxis_title="Falsas alarmas (tasa de falsos positivos)",
                      yaxis_title="Cancelaciones detectadas (recall)",
                      legend=dict(orientation="v", x=0.45, y=0.05, yanchor="bottom"))
    return fig


def _calibracion():
    c = S.csv("cancelacion/calibracion.csv")
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibración perfecta",
                    line=dict(dash="dot", color=T.GRIS_CLARO))
    fig.add_scatter(x=c["proba_media"], y=c["fraccion_real"], mode="lines+markers", name="Campeón",
                    line=dict(color=T.AZUL, width=3), marker=dict(size=9, color=T.AMARILLO,
                                                                  line=dict(color=T.AZUL_OSCURO, width=1)))
    fig.update_layout(title="Calibración: la probabilidad predicha coincide con la frecuencia real",
                      height=420, xaxis_title="Probabilidad predicha", yaxis_title="Proporción real de cancelaciones",
                      xaxis=dict(range=[0, 1], tickformat=".0%"), yaxis=dict(range=[0, 1], tickformat=".0%"))
    return fig


def layout():
    if not S.existe("cancelacion/metadata.json"):
        return html.Div([encabezado_pagina("4-5 · Modelado y evaluación", NOMBRE, ""), sin_artefactos("cancelacion")])
    meta = S.json_("cancelacion/metadata.json")
    m = meta["metricas"]
    comp = S.csv("cancelacion/comparacion_modelos.csv")
    imp = S.csv("cancelacion/importancia_shap.csv")
    graf_shap = [grafico(G.pareto_importancia(imp, meta["importancia"]["metodo"].split(" (")[0]))]
    if S.existe("cancelacion/shap_muestra.csv"):
        sel = imp.loc[imp["seleccionada"], "variable"].head(12).tolist()
        graf_shap.append(grafico(G.beeswarm_shap(S.csv("cancelacion/shap_muestra.csv"), sel)))
    mejora_brier = 100 * (1 - m["Brier"] / m["Brier_baseline"])
    return html.Div([
        encabezado_pagina(
            "4-5 · Modelado y evaluación", "¿Qué reservas se van a cancelar?",
            f"El campeón ({meta['campeon']['modelo']}, {meta['campeon']['n_variables']} variables) ordena "
            f"correctamente el riesgo en el {m['AUC_test'] * 100:.0f} % de los casos y entrega probabilidades "
            "confiables para decidir sobreventa, depósitos y recordatorios."),
        html.Div(className="fila-kpi", children=[
            kpi(f"{m['AUC_test']:.3f}".replace(".", ","), "AUC en test", "Capacidad de ordenar el riesgo", acento=True),
            kpi(f"{m['Brier']:.3f}".replace(".", ","), "Error de probabilidad (Brier)",
                f"{num(mejora_brier)} % menos que el baseline"),
            kpi(pct(100 * m["recall"]), "Cancelaciones detectadas", "Con umbral 0,5"),
            kpi(pct(100 * m["precision"]), "Precisión de las alertas", "Cuando predice «cancela»"),
            kpi(str(meta["campeon"]["n_variables"]), "Variables del campeón", meta["campeon"]["etapa"]),
        ]),
        nota(meta["justificacion"], "info"),
        html.H2("Simulador: probabilidad de cancelación de una reserva", className="titulo-seccion"),
        html.Div(className="rejilla-simulador", children=[
            tarjeta("Datos de la reserva", F.construir(PREFIJO, "cancelacion", EXCLUIR),
                    html.Button("Calcular probabilidad", id=f"{PREFIJO}-boton", className="boton")),
            tarjeta("Resultado", dcc.Loading(html.Div(id=f"{PREFIJO}-resultado",
                                                      children=nota("Completa los datos y presiona «Calcular».")),
                                             color=T.AZUL), clase="tarjeta-resultado"),
        ]),
        html.H2("¿Por qué ganó este modelo?", className="titulo-seccion"),
        tarjeta("Comparación de modelos: todas las variables vs. variables seleccionadas por SHAP", tabla_simple(
            comp, {"etapa": "Etapa", "modelo": "Modelo", "n_variables": "Variables", "AUC_test": "AUC",
                   "PR_AUC_test": "PR-AUC", "Brier": "Brier", "precision": "Precisión", "recall": "Recall",
                   "f1": "F1", "brecha_AUC": "Brecha AUC"},
            {c: (lambda x: "—" if x != x else f"{x:.3f}".replace(".", ","))
             for c in ["AUC_test", "PR_AUC_test", "Brier", "precision", "recall", "f1", "brecha_AUC"]}),
            nota("La fila resaltada es el campeón. Los modelos a menos del 1 % del mejor AUC se consideran "
                 "empatados y se elige el de menos variables (parsimonia). SVM no entrega probabilidades, por "
                 "eso no compite por el despliegue.")),
        tarjeta("Importancia de las variables", *graf_shap),
        html.Div(className="rejilla-2", children=[tarjeta(None, grafico(_roc())),
                                                  tarjeta(None, grafico(_calibracion()))]),
        tarjeta("Ajusta el umbral de decisión según el costo para el hotel", html.P(
            "Bajar el umbral detecta más cancelaciones a cambio de más falsas alarmas. Las cifras se recalculan "
            "sobre las reservas de prueba que el modelo no vio al entrenar.", className="ayuda"),
            dcc.Slider(id=f"{PREFIJO}-umbral", min=0.1, max=0.9, step=0.05, value=meta.get("umbral_decision", 0.5),
                       marks={v / 10: f"{v / 10:.1f}".replace(".", ",") for v in range(1, 10)},
                       tooltip={"placement": "bottom"}),
            html.Div(id=f"{PREFIJO}-umbral-salida")),
    ])


@callback(Output(f"{PREFIJO}-umbral-salida", "children"), Input(f"{PREFIJO}-umbral", "value"))
def actualizar_umbral(umbral):
    if umbral is None or not S.existe("cancelacion/probabilidades_test.csv"):
        return None
    r = S.metricas_umbral(umbral)
    z = [[r["VN"], r["FP"]], [r["FN"], r["VP"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Predice: no cancela", "Predice: cancela"],
                               y=["Real: no cancela", "Real: cancela"], colorscale=[[0, T.AZUL_CLARO], [1, T.AZUL]],
                               showscale=False, text=[[num(v) for v in fila] for fila in z],
                               texttemplate="%{text}", textfont=dict(size=18)))
    fig.update_layout(height=280, title=f"Matriz de confusión con umbral {umbral:.2f}".replace(".", ","),
                      yaxis=dict(autorange="reversed"))
    return html.Div(className="rejilla-2", children=[
        html.Div(className="fila-kpi fila-kpi-compacta", children=[
            kpi(pct(100 * r["recall"]), "Cancelaciones detectadas", f"{num(r['VP'])} de {num(r['VP'] + r['FN'])}"),
            kpi(pct(100 * r["precision"]), "Alertas correctas", f"{num(r['alertas'])} alertas"),
            kpi(num(r["FP"]), "Falsas alarmas", "Reservas que sí llegan"),
            kpi(num(r["FN"]), "Cancelaciones no anticipadas", "Habitaciones que se liberan sin aviso"),
        ]),
        grafico(fig),
    ])


@callback(Output(f"{PREFIJO}-resultado", "children"), Input(f"{PREFIJO}-boton", "n_clicks"),
          *F.estados(PREFIJO, EXCLUIR), prevent_initial_call=True)
def predecir(_, *args):
    try:
        r = S.predecir_cancelacion(F.valores(EXCLUIR, args))
    except Exception as e:  # noqa: BLE001
        return nota(f"No fue posible calcular la predicción: {e}", "alerta")
    from dashboard.graficos import gauge_probabilidad
    comparacion = "por encima" if r["proba"] > r["tasa_base"] else "por debajo"
    return html.Div([
        grafico(gauge_probabilidad(r["proba"], r["color"], r["tasa_base"])),
        html.Div(className="nivel-riesgo", style={"borderColor": r["color"]}, children=[
            html.Span(f"Riesgo {r['nivel'].lower()}", className="nivel-etiqueta",
                      style={"backgroundColor": r["color"], "color": "#1A1A1A" if r["nivel"] == "Medio" else "white"}),
            html.P([f"Probabilidad de cancelación de {pct(100 * r['proba'])}, ", html.B(comparacion),
                    f" de la tasa promedio ({pct(100 * r['tasa_base'])}, línea azul)."]),
            html.P([html.B("Acción sugerida: "), r["accion"]]),
            html.P("Con el umbral de decisión de " + f"{r['umbral']:.2f}".replace(".", ",") + " el modelo la "
                   "clasificaría como " + ("«cancela»." if r["proba"] >= r["umbral"] else "«no cancela»."),
                   className="ayuda"),
        ]),
    ])
