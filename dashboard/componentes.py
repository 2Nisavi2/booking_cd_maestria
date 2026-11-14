"""Componentes visuales reutilizables del tablero."""
from __future__ import annotations

from dash import dcc, html

from dashboard import tema


def encabezado_pagina(fase: str, titulo: str, mensaje: str):
    """Cabecera de cada página: fase CRISP-DM, título y mensaje titular (la conclusión)."""
    return html.Div(className="encabezado-pagina", children=[
        html.Span(fase, className="etiqueta-fase"),
        html.H1(titulo),
        html.P(mensaje, className="mensaje-titular"),
    ])


def kpi(valor: str, etiqueta: str, detalle: str | None = None, acento: bool = False):
    return html.Div(className="kpi" + (" kpi-acento" if acento else ""), children=[
        html.Div(valor, className="kpi-valor"),
        html.Div(etiqueta, className="kpi-etiqueta"),
        html.Div(detalle, className="kpi-detalle") if detalle else None,
    ])


def tarjeta(titulo: str | None, *hijos, clase: str = ""):
    return html.Div(className=f"tarjeta {clase}".strip(), children=[
        html.H3(titulo, className="tarjeta-titulo") if titulo else None, *hijos])


def grafico(fig, alto: int | None = None):
    if alto:
        fig.update_layout(height=alto)
    return dcc.Graph(figure=fig, config={"displayModeBar": False})


def nota(texto, tipo: str = "info"):
    return html.Div(texto, className=f"nota nota-{tipo}")


def tabla_simple(df, columnas: dict | None = None, formatos: dict | None = None):
    """Tabla HTML liviana. columnas: {col: encabezado}; formatos: {col: función}."""
    columnas = columnas or {c: c for c in df.columns}
    formatos = formatos or {}
    cabecera = html.Thead(html.Tr([html.Th(v) for v in columnas.values()]))
    filas = []
    for _, r in df.iterrows():
        celdas = []
        for c in columnas:
            val = r[c]
            celdas.append(html.Td(formatos[c](val) if c in formatos else val))
        filas.append(html.Tr(celdas, className="fila-destacada" if bool(r.get("campeon", False)) else None))
    return html.Div(html.Table([cabecera, html.Tbody(filas)], className="tabla"), className="tabla-contenedor")


def sin_artefactos(tarea: str):
    return nota(["Aún no hay artefactos para ", html.B(tarea), ". Ejecuta ",
                 html.Code(f"python main.py --tareas {tarea}"), " y vuelve a desplegar el tablero."],
                "alerta")


def num(x, dec: int = 0) -> str:
    """Formato numérico en español (miles con punto, decimales con coma)."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, dec: int = 1) -> str:
    return f"{num(x, dec)} %"


COLORES = tema
