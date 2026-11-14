"""Fase 6 · Despliegue: recomendaciones, limitaciones y próximos pasos."""
from dash import html

from dashboard import servicios as S
from dashboard.componentes import encabezado_pagina, nota, num, pct, tarjeta

RUTA = "/recomendaciones"
NOMBRE = "Recomendaciones"


def _d(x, dec=3):
    return f"{x:.{dec}f}".replace(".", ",")


def layout():
    hallazgos = []
    if S.existe("cancelacion/metadata.json"):
        m = S.json_("cancelacion/metadata.json")
        hallazgos.append(html.Li([html.B("Cancelación: "), f"{m['campeon']['modelo']} con "
                                  f"{m['campeon']['n_variables']} variables (AUC {_d(m['metricas']['AUC_test'])}, "
                                  f"Brier {_d(m['metricas']['Brier'])})."]))
    if S.existe("lead_time/metadata.json"):
        m = S.json_("lead_time/metadata.json")
        hallazgos.append(html.Li([html.B("Anticipación: "), f"{m['campeon']['modelo']} con error medio de "
                                  f"{num(m['metricas']['MAE_test'], 1)} días (R² {_d(m['metricas']['R2_test'], 2)})."]))
    if S.existe("segmentacion/metadata.json"):
        m = S.json_("segmentacion/metadata.json")
        hallazgos.append(html.Li([html.B("Segmentación: "), f"{m['campeon']['metodo']} con {m['campeon']['k']} "
                                  f"segmentos reproducibles con reglas ({pct(100 * m['metricas']['exactitud_arbol'])})."]))
    return html.Div([
        encabezado_pagina(
            "6 · Despliegue", "Del análisis a la decisión",
            "Cómo usar los modelos en la operación del hotel, qué tener presente y qué sigue."),
        html.Div(className="rejilla-3", children=[
            tarjeta("Probabilidad de cancelación", html.Ul([
                html.Li("Calcular a diario la probabilidad de cada reserva futura."),
                html.Li("Estimar cuántas habitaciones se liberarán por fecha y ajustar la sobreventa."),
                html.Li("Riesgo alto: pedir prepago o depósito y contactar al cliente."),
                html.Li("Riesgo medio: recordatorios y servicios adicionales que generen compromiso."),
            ])),
            tarjeta("Anticipación de la reserva", html.Ul([
                html.Li("Proyectar la curva de llenado de cada fecha de llegada."),
                html.Li("Definir tarifas según cuándo se espera que entren las reservas."),
                html.Li("Planear personal y campañas con la antelación de cada temporada."),
            ])),
            tarjeta("Segmentos", html.Ul([
                html.Li([html.B("Estancia larga: "), "depósitos y recordatorios (mayor cancelación)."]),
                html.Li([html.B("Negocios / fin de semana: "), "tarifas de último minuto y paquetes."]),
                html.Li([html.B("Premium: "), "servicios adicionales y venta cruzada."]),
                html.Li([html.B("Cortesía: "), "separar en los análisis de ingresos."]),
            ])),
        ]),
        html.Div(className="rejilla-2", children=[
            tarjeta("Resultados de los modelos desplegados", html.Ul(hallazgos) if hallazgos else
                    nota("Aún no hay modelos entrenados.", "alerta")),
            tarjeta("Limitaciones", html.Ul([
                html.Li("Datos de dos hoteles en Portugal entre 2015 y 2017: recalibrar antes de usarlos en otros contextos."),
                html.Li("La partición fue aleatoria y el año de llegada pesa en los modelos: validar con un corte temporal."),
                html.Li("Confirmar que los cambios en la reserva se conocen en el momento de la predicción."),
                html.Li("La probabilidad es una estimación de apoyo a la decisión, no una certeza."),
            ])),
        ]),
        tarjeta("Próximos pasos", html.Ol([
            html.Li("Validar los modelos supervisados con un corte temporal."),
            html.Li("Definir con el área comercial el costo de una cancelación no detectada y ajustar el umbral."),
            html.Li("Monitorear desempeño y calibración; reentrenar con python main.py cuando cambie el comportamiento."),
            html.Li("Incorporar un identificador único de reserva y la fecha de registro de cada dato."),
        ])),
    ])
