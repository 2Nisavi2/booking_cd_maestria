"""Fases 4-5 · Segmentación de reservas + asignador de segmento."""
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

from dashboard import servicios as S
from dashboard import tema as T
from dashboard.componentes import encabezado_pagina, grafico, kpi, nota, num, pct, sin_artefactos, tabla_simple, tarjeta

RUTA = "/segmentos"
NOMBRE = "Segmentos de clientes"
VARIABLES_Z = {"lead_time": "Antelación", "noches": "Noches", "prop_fin_semana": "% fin de semana",
               "huespedes": "Huéspedes", "con_ninos": "Con niños", "adr": "Tarifa", "solicitudes_especiales":
               "Solicitudes", "cambios_reserva": "Cambios", "resort": "Resort", "cancelada": "Cancelación",
               "huesped_repetido": "Repetido", "parqueadero": "Parqueadero"}


def _etiqueta(r):
    return f"S{int(r['segmento'])} · {r['nombre']}"


def _tamano_cancelacion(perfil):
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=("% de reservas", "Tasa de cancelación"))
    etiquetas = [_etiqueta(r) for _, r in perfil.iterrows()]
    fig.add_bar(y=etiquetas, x=perfil["pct"], orientation="h", marker_color=T.AZUL,
                text=[pct(v) for v in perfil["pct"]], textposition="outside", cliponaxis=False, row=1, col=1)
    mx = perfil["tasa_cancelacion_pct"].max()
    fig.add_bar(y=etiquetas, x=perfil["tasa_cancelacion_pct"], orientation="h",
                marker_color=[T.ROJO if v == mx else T.GRIS_CLARO for v in perfil["tasa_cancelacion_pct"]],
                text=[pct(v) for v in perfil["tasa_cancelacion_pct"]], textposition="outside", cliponaxis=False,
                row=1, col=2)
    fig.update_layout(showlegend=False, height=340, title="El segmento más grande es también el de mayor cancelación")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(showticklabels=False, showgrid=False)
    return fig


def _zscores(perfil):
    z = S.csv("segmentacion/zscores.csv").set_index("segmento")
    cols = [c for c in VARIABLES_Z if c in z.columns]
    nombres = {int(r["segmento"]): _etiqueta(r) for _, r in perfil.iterrows()}
    zz = z[cols].clip(-2, 2)
    fig = go.Figure(go.Heatmap(z=zz.values, x=[VARIABLES_Z[c] for c in cols], y=[nombres.get(i, i) for i in zz.index],
                               colorscale=[[0, T.AZUL], [0.5, "#FFFFFF"], [1, T.AMARILLO]], zmin=-2, zmax=2,
                               text=zz.round(1).values, texttemplate="%{text}", colorbar=dict(title="z")))
    fig.update_layout(title="Rasgos de cada segmento frente al promedio (amarillo: por encima; azul: por debajo)",
                      height=360, yaxis=dict(autorange="reversed"))
    return fig


def _pca(perfil):
    p = S.csv("segmentacion/proyeccion_pca.csv")
    meta = S.json_("segmentacion/metadata.json")
    var = meta.get("varianza_pca", [0, 0])
    nombres = {int(r["segmento"]): _etiqueta(r) for _, r in perfil.iterrows()}
    fig = go.Figure()
    for s, d in p.groupby("segmento"):
        fig.add_scatter(x=d["pc1"], y=d["pc2"], mode="markers", name=nombres.get(int(s), f"S{s}"),
                        marker=dict(size=4, opacity=0.5, color=T.SEGMENTOS[int(s) % len(T.SEGMENTOS)]))
    fig.update_layout(title="Segmentos en el plano de componentes principales", height=440,
                      xaxis_title=f"Componente 1 ({var[0]:.0%})", yaxis_title=f"Componente 2 ({var[1]:.0%})",
                      legend=dict(itemsizing="constant"))
    return fig


def _campo(id_, etiqueta, valor, mn, mx):
    return html.Div(className="campo", children=[
        html.Label(etiqueta, htmlFor=id_),
        dcc.Input(id=id_, type="number", value=valor, min=mn, max=mx, className="input-num", debounce=True)])


def layout():
    if not S.existe("segmentacion/metadata.json"):
        return html.Div([encabezado_pagina("4-5 · Segmentación", NOMBRE, ""), sin_artefactos("segmentacion")])
    meta = S.json_("segmentacion/metadata.json")
    m = meta["metricas"]
    perfil = S.csv("segmentacion/perfil_segmentos.csv")
    mayor = perfil.sort_values("tasa_cancelacion_pct", ascending=False).iloc[0]
    return html.Div([
        encabezado_pagina(
            "4-5 · Segmentación", f"{len(perfil)} perfiles de reservas para gestionar distinto",
            f"{meta['campeon']['metodo']} agrupa las reservas por antelación, duración, fin de semana, tarifa por "
            f"huésped y solicitudes. «{mayor['nombre']}» reúne el {pct(mayor['pct'])} de las reservas y cancela "
            f"el {pct(mayor['tasa_cancelacion_pct'])}."),
        html.Div(className="fila-kpi", children=[
            kpi(f"{meta['campeon']['metodo']} (k = {meta['campeon']['k']})", "Modelo campeón",
                "Mejor ranking en 3 métricas", acento=True),
            kpi(num(m["n_reservas"]), "Reservas segmentadas", "Únicas y válidas"),
            kpi(f"{m['silueta_global']:.2f}".replace(".", ","), "Silueta global", "Separación débil a moderada"),
            kpi(pct(100 * m["exactitud_arbol"]), "Reproducible con reglas", "Árbol de decisión de 4 niveles"),
        ]),
        html.Div(className="rejilla-2", children=[tarjeta(None, grafico(_tamano_cancelacion(perfil))),
                                                  tarjeta(None, grafico(_zscores(perfil)))]),
        tarjeta("Perfil de cada segmento", tabla_simple(
            perfil, {"segmento": "Segmento", "nombre": "Nombre sugerido", "pct": "% reservas",
                     "lead_time_mediana": "Antelación (mediana, días)", "noches_mediana": "Noches (mediana)",
                     "adr_mediana": "Tarifa (mediana, €)", "solicitudes_media": "Solicitudes (media)",
                     "tasa_cancelacion_pct": "% cancelación", "segmento_mercado_principal": "Canal principal"},
            {"segmento": lambda x: f"S{int(x)}", "pct": pct, "tasa_cancelacion_pct": pct,
             "lead_time_mediana": lambda x: num(x), "noches_mediana": lambda x: num(x),
             "adr_mediana": lambda x: num(x, 1), "solicitudes_media": lambda x: num(x, 2)})),
        html.H2("Asignador: ¿a qué segmento pertenece una reserva?", className="titulo-seccion"),
        html.Div(className="rejilla-simulador", children=[
            tarjeta("Comportamiento de la reserva", html.Div(className="rejilla-campos", children=[
                _campo("seg-lead", "Antelación (días)", 60, 0, 737),
                _campo("seg-finde", "Noches de fin de semana", 1, 0, 19),
                _campo("seg-semana", "Noches entre semana", 2, 0, 50),
                _campo("seg-adr", "Tarifa diaria (€)", 95, 0, 999),
                _campo("seg-huespedes", "Huéspedes", 2, 1, 10),
                _campo("seg-solicitudes", "Solicitudes especiales", 0, 0, 5),
            ]), html.Button("Asignar segmento", id="seg-boton", className="boton")),
            tarjeta("Resultado", html.Div(id="seg-resultado", children=nota("Ingresa los datos y presiona «Asignar».")),
                    clase="tarjeta-resultado"),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta(None, grafico(_pca(perfil))),
            tarjeta("Reglas de negocio que reproducen los segmentos",
                    html.P("Árbol de decisión de 4 niveles sobre las variables en su escala original "
                           "(class = segmento asignado).", className="ayuda"),
                    html.Pre(meta["reglas_arbol"], className="reglas")),
        ]),
        nota("La separación entre grupos es moderada: el comportamiento de las reservas es un continuo y los "
             "segmentos son una partición útil para la gestión, no grupos «naturales».", "alerta"),
    ])


@callback(Output("seg-resultado", "children"), Input("seg-boton", "n_clicks"),
          State("seg-lead", "value"), State("seg-finde", "value"), State("seg-semana", "value"),
          State("seg-adr", "value"), State("seg-huespedes", "value"), State("seg-solicitudes", "value"),
          prevent_initial_call=True)
def asignar(_, lead, finde, semana, adr, huespedes, solicitudes):
    valores = [lead, finde, semana, adr, huespedes, solicitudes]
    if any(v is None for v in valores):
        return nota("Completa todos los campos con valores válidos.", "alerta")
    try:
        r = S.asignar_segmento(*valores)
    except Exception as e:  # noqa: BLE001
        return nota(f"No fue posible asignar el segmento: {e}", "alerta")
    p = r["perfil"]
    color = T.SEGMENTOS[r["segmento"] % len(T.SEGMENTOS)]
    return html.Div([
        html.Div(className="resultado-grande", style={"borderLeftColor": color}, children=[
            html.Div(f"S{r['segmento']}", className="resultado-valor"),
            html.Div(r["nombre"], className="resultado-etiqueta"),
        ]),
        html.P(f"Este segmento reúne el {pct(p.get('pct', 0))} de las reservas, con antelación mediana de "
               f"{num(p.get('lead_time_mediana', 0))} días, tarifa mediana de {num(p.get('adr_mediana', 0), 1)} € "
               f"y {pct(p.get('tasa_cancelacion_pct', 0))} de cancelación.") if p else None,
        nota([html.B("Acción sugerida: "), r["accion"]], "ok"),
    ])
