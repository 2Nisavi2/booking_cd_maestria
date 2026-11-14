"""Segmentación de reservas por comportamiento (K-Means, GMM, Ward, Aglomerativo, DBSCAN)."""
from __future__ import annotations

import itertools
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from src.logging_config import log_etapa
from src.segmentos import (EQUIVALENCIA_ORIGINAL, VARIABLES_AGRUPACION, AsignadorSegmentos,
                           variables_comportamiento)
from training import artefactos

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1) Base de segmentación
# --------------------------------------------------------------------------
def preparar_base_segmentacion(df: pd.DataFrame, variables: list | None = None) -> dict:
    d = df.drop_duplicates()
    n0 = len(d)
    guests = d["adults"] + d["children"].fillna(0) + d["babies"]
    d = d[(guests > 0) & (d["adr"] >= 0) & (d["adr"] < 1000)].copy()  # reservas inválidas / atípicas
    guests = d["adults"] + d["children"].fillna(0) + d["babies"]
    noches = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
    perfil = pd.DataFrame({
        "lead_time": d["lead_time"], "noches": noches,
        "prop_fin_semana": np.where(noches > 0, d["stays_in_weekend_nights"] / noches.replace(0, 1), 0),
        "huespedes": guests, "con_ninos": ((d["children"].fillna(0) + d["babies"]) > 0).astype(int),
        "adr": d["adr"], "adr_por_huesped": d["adr"] / guests,
        "solicitudes_especiales": d["total_of_special_requests"],
        "cambios_reserva": d["booking_changes"], "resort": (d["hotel"] == "Resort Hotel").astype(int),
        "cancelada": d["is_canceled"], "huesped_repetido": d["is_repeated_guest"],
        "parqueadero": (d["required_car_parking_spaces"] > 0).astype(int),
        "dias_lista_espera": d["days_in_waiting_list"],
        "hotel": d["hotel"], "market_segment": d["market_segment"],
        "distribution_channel": d["distribution_channel"], "customer_type": d["customer_type"],
        "deposit_type": d["deposit_type"], "meal": d["meal"],
        "pais": np.where(d["country"] == "PRT", "Portugal", "Extranjero"),
        "mes_llegada": d["arrival_date_month"],
    }, index=d.index)
    variables = variables or VARIABLES_AGRUPACION
    X = variables_comportamiento(perfil)[variables]
    escalador = StandardScaler().fit(X)
    Xs = pd.DataFrame(escalador.transform(X), columns=X.columns, index=X.index)
    X_original = perfil[[EQUIVALENCIA_ORIGINAL[v] for v in variables]]
    logger.info("Base de segmentación: %d -> %d sin duplicados -> %d reservas válidas | "
                "variables de agrupación: %s", len(df), n0, len(d), ", ".join(variables))
    return {"X": X, "X_esc": Xs, "X_original": X_original, "perfil": perfil,
            "escalador": escalador, "variables": variables}


# --------------------------------------------------------------------------
# 2) Entrenamiento y comparación
# --------------------------------------------------------------------------
def _metricas(Z, etiquetas):
    mask = etiquetas >= 0  # DBSCAN: excluye ruido
    lab = etiquetas[mask]
    k = len(np.unique(lab))
    if k < 2:
        return dict(k=k, silueta=np.nan, davies_bouldin=np.nan, calinski=np.nan,
                    menor_cluster_pct=np.nan, ruido_pct=100 * (1 - mask.mean()))
    tamanos = np.unique(lab, return_counts=True)[1]
    return dict(k=k, silueta=silhouette_score(Z[mask], lab),
                davies_bouldin=davies_bouldin_score(Z[mask], lab),
                calinski=calinski_harabasz_score(Z[mask], lab),
                menor_cluster_pct=100 * tamanos.min() / len(etiquetas),
                ruido_pct=100 * (1 - mask.mean()))


def entrenar_modelos_segmentacion(base, k_min=2, k_max=8, n_muestra=8000, min_cluster_pct=1.5,
                                  max_ruido_pct=10.0, random_state=42) -> dict:
    Xs = base["X_esc"]
    muestra = Xs.sample(min(n_muestra, len(Xs)), random_state=random_state)
    Z = muestra.to_numpy()
    filas, etiquetas_muestra = [], {}
    logger.info("Muestra de entrenamiento: %d reservas | k = %d..%d | restricciones: cluster >= %.1f %% "
                "y ruido <= %.0f %%", len(muestra), k_min, k_max, min_cluster_pct, max_ruido_pct)

    def registrar(metodo, param, etiquetas, extra=None, segundos=0.0):
        m = _metricas(Z, etiquetas)
        m.update({"metodo": metodo, "parametro": param, "segundos": segundos})
        if extra:
            m.update(extra)
        filas.append(m)
        etiquetas_muestra[(metodo, param)] = etiquetas

    rango = range(k_min, k_max + 1)
    logger.info("[1/5] K-Means...")
    for k in rango:
        t = time.time()
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        registrar("K-Means", f"k={k}", km.labels_, {"inercia": km.inertia_}, time.time() - t)
    logger.info("[2/5] Mixtura Gaussiana...")
    for k in rango:
        t = time.time()
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                             random_state=random_state).fit(Z)
        registrar("Mixtura Gaussiana", f"k={k}", gm.predict(Z), {"BIC": gm.bic(Z)}, time.time() - t)
    logger.info("[3/5] Jerárquico (Ward)...")
    t = time.time()
    arbol = linkage(Z, method="ward")
    for k in rango:
        registrar("Jerárquico (Ward)", f"k={k}", fcluster(arbol, k, criterion="maxclust") - 1,
                  segundos=(time.time() - t) / len(rango))
    logger.info("[4/5] Aglomerativo (enlace promedio)...")
    for k in rango:
        t = time.time()
        ag = AgglomerativeClustering(n_clusters=k, linkage="average").fit(Z)
        registrar("Aglomerativo (promedio)", f"k={k}", ag.labels_, segundos=time.time() - t)
    logger.info("[5/5] DBSCAN (eps desde la curva de k-distancias)...")
    min_samples = 2 * Z.shape[1]
    dist = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)[0][:, -1])
    for q in [0.80, 0.85, 0.90, 0.95, 0.98]:
        eps = float(np.quantile(dist, q))
        t = time.time()
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(Z)
        registrar("DBSCAN", f"eps={eps:.2f}", db.labels_, {"eps": eps}, time.time() - t)

    tabla = pd.DataFrame(filas)
    tabla["valida"] = (tabla["k"] >= 2) & (tabla["menor_cluster_pct"] >= min_cluster_pct) & \
                      (tabla["ruido_pct"] <= max_ruido_pct)
    mejores = (tabla[tabla["valida"]].sort_values("silueta", ascending=False)
               .groupby("metodo").head(1).copy())
    if mejores.empty:
        raise ValueError("Ninguna configuración cumple las restricciones; relaja min_cluster_pct.")
    mejores["rank"] = (mejores["silueta"].rank(ascending=False)
                       + mejores["davies_bouldin"].rank(ascending=True)
                       + mejores["calinski"].rank(ascending=False)) / 3
    mejores = mejores.sort_values(["rank", "silueta"], ascending=[True, False]).reset_index(drop=True)
    campeon = mejores.iloc[0]
    for _, r in mejores.iterrows():
        logger.info("   %-24s %-8s silueta=%.3f | DB=%.3f | CH=%.0f | ranking=%.2f", r["metodo"],
                    r["parametro"], r["silueta"], r["davies_bouldin"], r["calinski"], r["rank"])
    sin_config = sorted(set(tabla["metodo"]) - set(mejores["metodo"]))
    if sin_config:
        logger.info("Sin configuración válida: %s (clusters diminutos o demasiado ruido)",
                    ", ".join(sin_config))
    logger.info("🏆 Modelo campeón: %s (%s) | silueta = %.3f", campeon["metodo"], campeon["parametro"],
                campeon["silueta"])
    return {"tabla": tabla, "mejores": mejores, "campeon": campeon, "muestra": muestra,
            "etiquetas_muestra": etiquetas_muestra, "random_state": random_state}


def asignar_segmentos(base, res_modelos):
    """Asigna un segmento a TODAS las reservas y construye el asignador para el tablero."""
    campeon = res_modelos["campeon"]
    metodo, parametro = campeon["metodo"], campeon["parametro"]
    Xs = base["X_esc"]
    rs = res_modelos["random_state"]
    fila = res_modelos["tabla"].query("metodo == @metodo and parametro == @parametro").iloc[0]
    k = int(fila["k"])
    if metodo == "K-Means":
        modelo = KMeans(n_clusters=k, n_init=10, random_state=rs).fit(Xs.to_numpy())
        etiquetas = modelo.labels_
    elif metodo == "Mixtura Gaussiana":
        modelo = GaussianMixture(n_components=k, covariance_type="full", n_init=3,
                                 random_state=rs).fit(Xs.to_numpy())
        etiquetas = modelo.predict(Xs.to_numpy())
    else:
        lab = res_modelos["etiquetas_muestra"][(metodo, parametro)]
        Zm = res_modelos["muestra"].to_numpy()
        core = lab >= 0
        modelo = KNeighborsClassifier(n_neighbors=15).fit(Zm[core], lab[core])
        etiquetas = modelo.predict(Xs.to_numpy())
    etiquetas = pd.Series(etiquetas, index=Xs.index, name="segmento")
    orden = etiquetas[etiquetas >= 0].value_counts().index
    mapa = {int(old): new for new, old in enumerate(orden)}
    mapa[-1] = -1
    segmentos = etiquetas.map(mapa).astype(int)
    asignador = AsignadorSegmentos(base["variables"], base["escalador"], modelo, mapa)
    tam = segmentos.value_counts(normalize=True).sort_index() * 100
    logger.info("Segmentos asignados a %d reservas: %s", len(segmentos),
                " | ".join(f"S{s}: {p:.1f} %" for s, p in tam.items()))
    return segmentos, asignador


# --------------------------------------------------------------------------
# 3) Perfiles y nombres
# --------------------------------------------------------------------------
def _cliffs_delta(a, b, max_n=3000, rng=np.random.default_rng(0)):
    a = rng.choice(a, min(len(a), max_n), replace=False)
    b = rng.choice(b, min(len(b), max_n), replace=False)
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


def nombrar_segmentos(medianas: pd.DataFrame, medias: pd.DataFrame) -> dict:
    """Asigna nombres de negocio a los segmentos según sus rasgos (heurística del informe)."""
    libres = list(medianas.index)
    nombres = {}

    def tomar(nombre, criterio, ascendente=False):
        if not libres:
            return
        s = criterio.loc[libres].sort_values(ascending=ascendente).index[0]
        nombres[int(s)] = nombre
        libres.remove(s)

    # Orden de prioridad: rasgos más inequívocos primero
    if (medianas.loc[libres, "adr"] < 20).any():
        tomar("Estancias de cortesía / clientes internos", medianas["adr"], ascendente=True)
    tomar("Escapadas de fin de semana", medias["prop_fin_semana"])
    tomar("Vacacionistas exigentes / premium", medias["solicitudes_especiales"] + medias["adr"] / 100)
    tomar("Viajeros de negocios / última hora", medianas["lead_time"], ascendente=True)
    tomar("Planificadores de estancia larga", medianas["lead_time"] + medianas["noches"] * 10)
    for s in libres:
        nombres[int(s)] = f"Segmento S{s}"
    return dict(sorted(nombres.items()))


def perfilar_segmentos(base, segmentos, alpha=0.05, n_silueta=10_000, random_state=42) -> dict:
    perfil = base["perfil"].loc[segmentos.index].copy()
    perfil["segmento"] = segmentos
    validos = perfil[perfil["segmento"] >= 0]
    segs = sorted(validos["segmento"].unique())
    num = ["lead_time", "noches", "prop_fin_semana", "huespedes", "con_ninos", "adr", "adr_por_huesped",
           "solicitudes_especiales", "cambios_reserva", "resort", "cancelada", "huesped_repetido",
           "parqueadero", "dias_lista_espera"]
    cat = ["hotel", "market_segment", "distribution_channel", "customer_type", "deposit_type", "meal",
           "pais", "mes_llegada"]

    tam = perfil["segmento"].value_counts().sort_index().rename("n").to_frame()
    tam["%"] = (tam["n"] / tam["n"].sum() * 100).round(1)
    medias = validos.groupby("segmento")[num].mean()
    medianas = validos.groupby("segmento")[num].median()
    z = (medias - validos[num].mean()) / validos[num].std()

    logger.info("Pruebas estadísticas entre segmentos (Kruskal-Wallis / chi2)...")
    pruebas = []
    n_pares = len(list(itertools.combinations(segs, 2)))
    for v in num:
        grupos = [validos.loc[validos["segmento"] == s, v].to_numpy() for s in segs]
        H, p = stats.kruskal(*grupos)
        eps2 = (H - len(segs) + 1) / (len(validos) - len(segs))
        sig = 0
        for a, b in itertools.combinations(range(len(segs)), 2):
            p_ab = stats.mannwhitneyu(grupos[a], grupos[b]).pvalue
            if p_ab * n_pares < alpha and abs(_cliffs_delta(grupos[a], grupos[b])) >= 0.147:
                sig += 1
        pruebas.append({"variable": v, "tipo": "numérica", "prueba": "Kruskal-Wallis", "estadistico": H,
                        "p_valor": p, "efecto": eps2, "medida": "épsilon²",
                        "pares_distintos_%": 100 * sig / n_pares})
    for v in cat:
        tabla = pd.crosstab(validos[v], validos["segmento"])
        chi2, p, _, _ = stats.chi2_contingency(tabla)
        cv = np.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))
        pruebas.append({"variable": v, "tipo": "categórica", "prueba": "chi2", "estadistico": chi2,
                        "p_valor": p, "efecto": cv, "medida": "V de Cramér", "pares_distintos_%": np.nan})
    pruebas = pd.DataFrame(pruebas).sort_values("efecto", ascending=False).reset_index(drop=True)

    composicion = {}
    for v in cat:
        composicion[v] = (pd.crosstab(validos["segmento"], validos[v], normalize="index") * 100).round(1)

    nombres = nombrar_segmentos(medianas, medias)
    descripcion = pd.DataFrame({
        "segmento": segs, "nombre": [nombres[s] for s in segs],
        "n": tam.loc[segs, "n"].values, "pct": tam.loc[segs, "%"].values,
        "lead_time_mediana": medianas["lead_time"].round(0).values,
        "noches_mediana": medianas["noches"].values,
        "adr_mediana": medianas["adr"].round(1).values,
        "solicitudes_media": medias["solicitudes_especiales"].round(2).values,
        "prop_fin_semana_media": medias["prop_fin_semana"].round(3).values,
        "con_ninos_pct": (medias["con_ninos"] * 100).round(1).values,
        "tasa_cancelacion_pct": (medias["cancelada"] * 100).round(1).values,
        "resort_pct": (medias["resort"] * 100).round(1).values,
        "segmento_mercado_principal": composicion["market_segment"].idxmax(axis=1).loc[segs].values,
        "portugal_pct": composicion["pais"].get("Portugal", pd.Series(0, index=segs)).loc[segs].values,
    })
    for _, r in descripcion.iterrows():
        logger.info("   S%d %-42s %5.1f %% | lead_time med. %3.0f d | noches %.0f | ADR %.0f € | "
                    "cancelación %.1f %%", r["segmento"], r["nombre"], r["pct"], r["lead_time_mediana"],
                    r["noches_mediana"], r["adr_mediana"], r["tasa_cancelacion_pct"])

    logger.info("Nitidez de fronteras (silueta) y reglas con árbol de decisión...")
    Xs = base["X_esc"].loc[validos.index]
    m = Xs.sample(min(n_silueta, len(Xs)), random_state=random_state)
    lab_m = validos.loc[m.index, "segmento"].to_numpy()
    sil = silhouette_samples(m.to_numpy(), lab_m)
    fronteras = pd.DataFrame({"segmento": lab_m, "silueta": sil}).groupby("segmento")["silueta"].agg(
        silueta_media="mean",
        mal_ubicadas_pct=lambda s: 100 * (s < 0).mean(),
        frontera_difusa_pct=lambda s: 100 * ((s >= 0) & (s < 0.1)).mean(),
        nucleo_claro_pct=lambda s: 100 * (s > 0.5).mean()).round(3).reset_index()
    arbol = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=random_state)
    Xo = base["X_original"].loc[validos.index]
    exactitud = cross_val_score(arbol, Xo, validos["segmento"], cv=5).mean()
    arbol.fit(Xo, validos["segmento"])
    reglas = export_text(arbol, feature_names=list(Xo.columns), decimals=2)
    importancia_arbol = pd.Series(arbol.feature_importances_, index=Xo.columns) \
        .sort_values(ascending=False).round(3)
    logger.info("Silueta global: %.3f | mal ubicadas: %.1f %% | árbol (prof. 4) reproduce los "
                "segmentos con exactitud CV = %.1f %%", sil.mean(), 100 * (sil < 0).mean(), 100 * exactitud)

    pca = PCA(n_components=2, random_state=random_state).fit(base["X_esc"])
    P = pca.transform(m.iloc[:4000])
    proyeccion = pd.DataFrame({"pc1": P[:, 0], "pc2": P[:, 1], "segmento": lab_m[:4000]})

    return {"descripcion": descripcion, "medias": medias, "z": z, "pruebas": pruebas,
            "composicion": composicion, "fronteras": fronteras, "silueta_global": float(sil.mean()),
            "mal_ubicadas_pct": float(100 * (sil < 0).mean()), "exactitud_arbol": float(exactitud),
            "reglas": reglas, "importancia_arbol": importancia_arbol, "nombres": nombres,
            "proyeccion": proyeccion, "varianza_pca": pca.explained_variance_ratio_.tolist()}


def ejecutar(df_new: pd.DataFrame, cfg: dict) -> dict:
    c, g = cfg["segmentacion"], cfg["general"]
    salida = g["artefactos"] / "segmentacion"

    with log_etapa(logger, "Preparación de la base de segmentación"):
        base = preparar_base_segmentacion(df_new)
    with log_etapa(logger, "Entrenamiento y comparación de 5 algoritmos de clustering"):
        res = entrenar_modelos_segmentacion(base, c["k_min"], c["k_max"], c["n_muestra"],
                                            c["min_cluster_pct"], c["max_ruido_pct"], g["semilla"])
    with log_etapa(logger, "Asignación de segmentos a todas las reservas"):
        segmentos, asignador = asignar_segmentos(base, res)
    with log_etapa(logger, "Perfilamiento estadístico de los segmentos"):
        perf = perfilar_segmentos(base, segmentos, n_silueta=c["n_silueta"], random_state=g["semilla"])
        asignador.nombres = perf["nombres"]

    with log_etapa(logger, "Exportación de artefactos (segmentación)"):
        artefactos.guardar_modelo(asignador, salida / "asignador.joblib")
        artefactos.guardar_tabla(res["tabla"].drop(columns=[], errors="ignore"), salida / "comparacion_configuraciones.csv")
        artefactos.guardar_tabla(res["mejores"], salida / "mejores_por_metodo.csv")
        artefactos.guardar_tabla(perf["descripcion"], salida / "perfil_segmentos.csv")
        artefactos.guardar_tabla(perf["z"].round(3).reset_index(), salida / "zscores.csv")
        artefactos.guardar_tabla(perf["pruebas"].round(5), salida / "pruebas_estadisticas.csv")
        artefactos.guardar_tabla(perf["fronteras"], salida / "fronteras.csv")
        artefactos.guardar_tabla(perf["proyeccion"].round(4), salida / "proyeccion_pca.csv")
        artefactos.guardar_tabla(perf["composicion"]["market_segment"].reset_index(),
                                 salida / "composicion_mercado.csv")
        campeon = res["campeon"]
        meta = {"tarea": "segmentacion", "campeon": {"metodo": campeon["metodo"],
                                                     "parametro": campeon["parametro"],
                                                     "k": int(campeon["k"])},
                "variables": base["variables"], "nombres": {str(k): v for k, v in perf["nombres"].items()},
                "metricas": {"silueta_muestra": float(campeon["silueta"]),
                             "davies_bouldin": float(campeon["davies_bouldin"]),
                             "calinski": float(campeon["calinski"]),
                             "silueta_global": perf["silueta_global"],
                             "mal_ubicadas_pct": perf["mal_ubicadas_pct"],
                             "exactitud_arbol": perf["exactitud_arbol"],
                             "n_reservas": int(len(segmentos))},
                "reglas_arbol": perf["reglas"],
                "importancia_arbol": perf["importancia_arbol"].to_dict(),
                "varianza_pca": perf["varianza_pca"],
                "modo_rapido": bool(g["rapido"]),
                "versiones": artefactos._versiones()}
        artefactos.guardar_json(meta, salida / "metadata.json")
    return {"tarea": "segmentacion", "campeon": f"{campeon['metodo']} ({campeon['parametro']})",
            "silueta": round(perf["silueta_global"], 3), "exactitud_arbol": round(perf["exactitud_arbol"], 3)}
