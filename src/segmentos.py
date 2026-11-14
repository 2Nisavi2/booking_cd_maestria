"""Utilidades de segmentación compartidas por el entrenamiento y el tablero."""
from __future__ import annotations

import numpy as np
import pandas as pd

VARIABLES_AGRUPACION = ["lead_time_log", "noches_log", "prop_fin_semana", "adr_huesped_log",
                        "solicitudes"]
EQUIVALENCIA_ORIGINAL = {"lead_time_log": "lead_time", "noches_log": "noches",
                         "prop_fin_semana": "prop_fin_semana", "huespedes": "huespedes",
                         "con_ninos": "con_ninos", "adr_huesped_log": "adr_por_huesped",
                         "solicitudes": "solicitudes_especiales", "cambios": "cambios_reserva",
                         "resort": "resort"}


def variables_comportamiento(perfil: pd.DataFrame) -> pd.DataFrame:
    """Variables de agrupación (transformadas) a partir del perfil en escala original."""
    return pd.DataFrame({
        "lead_time_log": np.log1p(perfil["lead_time"]),
        "noches_log": np.log1p(perfil["noches"]),
        "prop_fin_semana": perfil["prop_fin_semana"],
        "huespedes": perfil["huespedes"].clip(1, 5),
        "con_ninos": perfil["con_ninos"],
        "adr_huesped_log": np.log1p(perfil["adr_por_huesped"]),
        "solicitudes": perfil["solicitudes_especiales"].clip(0, 3),
        "cambios": perfil["cambios_reserva"].clip(0, 3),
        "resort": perfil["resort"],
    }, index=perfil.index)


class AsignadorSegmentos:
    """Asigna el segmento (renumerado por tamaño) a reservas nuevas.

    Envuelve el escalador y el modelo campeón (K-Means / GMM) o, para métodos
    sin `predict` (jerárquicos), un KNN entrenado sobre la muestra etiquetada.
    """

    def __init__(self, variables, escalador, modelo, mapa, nombres=None):
        self.variables = list(variables)
        self.escalador = escalador
        self.modelo = modelo
        self.mapa = dict(mapa)
        self.nombres = dict(nombres or {})

    def predecir(self, perfil: pd.DataFrame) -> np.ndarray:
        X = variables_comportamiento(perfil)[self.variables]
        Z = self.escalador.transform(X)
        crudos = self.modelo.predict(Z)
        return np.array([self.mapa.get(int(c), -1) for c in crudos])

    def predecir_reserva(self, lead_time, noches_fin_semana, noches_semana, adr, huespedes,
                         solicitudes, ninos=0, cambios=0, resort=0) -> int:
        noches = noches_fin_semana + noches_semana
        huespedes = max(int(huespedes), 1)
        perfil = pd.DataFrame([{
            "lead_time": lead_time, "noches": noches,
            "prop_fin_semana": noches_fin_semana / noches if noches > 0 else 0,
            "huespedes": huespedes, "con_ninos": int(ninos > 0),
            "adr_por_huesped": max(adr, 0) / huespedes, "solicitudes_especiales": solicitudes,
            "cambios_reserva": cambios, "resort": resort}])
        return int(self.predecir(perfil)[0])
