"""Tablero de storytelling y predicción · Proyecto Hotel Booking Demand.

Local:      python -m dashboard.app          (http://127.0.0.1:8050)
Producción: gunicorn dashboard.app:server     (ver Procfile)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from dash import Dash, Input, Output, dcc, html  # noqa: E402

from dashboard import servicios as S  # noqa: E402,F401
from dashboard import tema  # noqa: E402,F401  (registra la plantilla de gráficos)
from dashboard.pages import (anticipacion, cancelacion, datos, factores, inicio,  # noqa: E402
                             recomendaciones, segmentos)

PAGINAS = [  # (módulo, fase CRISP-DM)
    (inicio, "1 · Negocio"),
    (datos, "2-3 · Datos"),
    (factores, "2 · Descriptivo"),
    (cancelacion, "4-5 · Modelo"),
    (anticipacion, "4-5 · Modelo"),
    (segmentos, "4-5 · Modelo"),
    (recomendaciones, "6 · Despliegue"),
]
RUTAS = {m.RUTA: m for m, _ in PAGINAS}

app = Dash(__name__, title="Demanda hotelera · Proyecto académico", suppress_callback_exceptions=True,
           update_title=None, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server  # para gunicorn

for _aviso in S.verificar_versiones():
    print(f"[AVISO de versiones] {_aviso}. Ajusta requirements.txt para evitar diferencias.", flush=True)


def _navegacion():
    return html.Nav(className="barra-lateral", children=[
        html.Div(className="marca", children=[
            html.Span("Demanda Hotelera", className="marca-logo"),
            html.Span("Proyecto Hotel Booking Demand · CRISP-DM", className="marca-sub"),
        ]),
        html.Div(className="menu", children=[
            dcc.Link(href=m.RUTA, className="enlace-menu", id=f"nav-{m.RUTA.strip('/') or 'inicio'}", children=[
                html.Span(fase, className="enlace-fase"), html.Span(m.NOMBRE, className="enlace-nombre")])
            for m, fase in PAGINAS
        ]),
        html.Div(className="pie-menu", children=[
            html.P("Diego Nicolás Ávila Moreno"),
            html.P("Maestría en IA · Universidad de La Salle"),
        ]),
    ])


app.layout = html.Div(className="contenedor", children=[
    dcc.Location(id="url"),
    _navegacion(),
    html.Main(className="contenido", children=[
        html.Div(id="pagina"),
        html.Footer("Datos: Hotel Booking Demand (António, Almeida y Nunes, 2019) · Kaggle · CC BY 4.0. "
                    "Las predicciones son estimaciones de apoyo a la decisión.", className="pie"),
    ]),
])


@app.callback(Output("pagina", "children"), *[Output(f"nav-{m.RUTA.strip('/') or 'inicio'}", "className")
                                              for m, _ in PAGINAS], Input("url", "pathname"))
def enrutar(ruta):
    modulo = RUTAS.get(ruta or "/", inicio)
    clases = ["enlace-menu activo" if m is modulo else "enlace-menu" for m, _ in PAGINAS]
    return modulo.layout(), *clases


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)),
            debug=os.environ.get("DASH_DEBUG", "0") == "1")
