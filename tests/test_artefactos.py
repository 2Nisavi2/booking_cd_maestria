"""Pruebas de humo de los artefactos exportados: se omiten si aún no se ha entrenado."""
import pytest

from dashboard import servicios as S

requiere = lambda t: pytest.mark.skipif(not S.existe(f"{t}/metadata.json"),  # noqa: E731
                                        reason=f"sin artefactos de {t}")


@requiere("cancelacion")
def test_prediccion_cancelacion_es_probabilidad():
    r = S.predecir_cancelacion({"fecha_llegada": "2016-08-15", "lead_time": 120})
    assert 0 <= r["proba"] <= 1
    assert r["nivel"] in {"Bajo", "Medio", "Alto"}


@requiere("cancelacion")
def test_deposito_no_reembolsable_aumenta_riesgo():
    base = {"fecha_llegada": "2016-08-15", "lead_time": 200, "market_segment": "Groups",
            "distribution_channel": "TA/TO"}
    sin = S.predecir_cancelacion({**base, "deposit_type": "No Deposit"})["proba"]
    con = S.predecir_cancelacion({**base, "deposit_type": "Non Refund"})["proba"]
    assert con >= sin


@requiere("cancelacion")
def test_metricas_umbral_consistentes():
    r = S.metricas_umbral(0.5)
    assert r["VP"] + r["FP"] + r["FN"] + r["VN"] == r["total"]
    assert S.metricas_umbral(0.2)["recall"] >= S.metricas_umbral(0.8)["recall"]


@requiere("lead_time")
def test_prediccion_lead_time_no_negativa():
    r = S.predecir_lead_time({"fecha_llegada": "2017-07-20"})
    assert r["dias"] >= 0


@requiere("segmentacion")
def test_asignador_devuelve_segmento_valido():
    r = S.asignar_segmento(120, 2, 5, 90, 2, 0)
    assert r["segmento"] >= 0 and r["nombre"]
