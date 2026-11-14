"""Paleta y plantilla de gráficos inspiradas en la identidad visual de Booking.

Contrastes verificados para texto: blanco sobre AZUL_OSCURO (12,6:1) y sobre
AZUL (4,9:1); texto oscuro sobre AMARILLO (12:1).
"""
import plotly.graph_objects as go
import plotly.io as pio

AZUL_OSCURO = "#003580"   # cabecera y navegación
AZUL = "#006CE4"          # serie principal, botones, enlaces
AZUL_CLARO = "#EBF3FF"    # fondos suaves
AZUL_MEDIO = "#5B9BE8"    # serie secundaria
AMARILLO = "#FEBB02"      # acento / resaltado
TEXTO = "#1A1A1A"
GRIS = "#595959"
GRIS_CLARO = "#A3A3A3"
BORDE = "#E7E7E7"
FONDO = "#F5F7FA"
ROJO = "#D4111E"          # riesgo alto
VERDE = "#008009"         # riesgo bajo / positivo

SECUENCIA = [AZUL, AZUL_OSCURO, AMARILLO, AZUL_MEDIO, GRIS_CLARO, "#0A8A8A", "#B54708"]
HOTEL = {"City Hotel": AZUL, "Resort Hotel": AMARILLO}
SEGMENTOS = [AZUL_OSCURO, AZUL, AMARILLO, AZUL_MEDIO, GRIS_CLARO, "#0A8A8A", "#B54708", "#6B4FBB"]

FUENTE = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

pio.templates["booking"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FUENTE, size=13, color=TEXTO),
        colorway=SECUENCIA,
        paper_bgcolor="white", plot_bgcolor="white",
        # Título arriba del todo (referido al contenedor, no al área del gráfico)
        title=dict(font=dict(size=15, color=TEXTO), x=0, xanchor="left", xref="paper",
                   y=0.98, yanchor="top", yref="container"),
        # automargin: Plotly amplía el margen para que etiquetas y títulos de ejes no se recorten
        xaxis=dict(showgrid=False, linecolor=BORDE, ticks="outside", tickcolor=BORDE, zeroline=False,
                   automargin=True, title=dict(standoff=12)),
        yaxis=dict(gridcolor="#EEF0F3", linecolor=BORDE, zeroline=False, automargin=True,
                   title=dict(standoff=12)),
        # Leyenda en la franja entre el título y el área del gráfico (margen superior amplio)
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
        margin=dict(l=20, r=30, t=95, b=20, pad=4),
        hoverlabel=dict(bgcolor="white", bordercolor=AZUL, font=dict(color=TEXTO)),
    )
)
pio.templates.default = "booking"


def fig_base(fig: go.Figure, titulo: str | None = None, alto: int = 360) -> go.Figure:
    fig.update_layout(height=alto, title_text=titulo)
    return fig
