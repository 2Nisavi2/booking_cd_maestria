# Modelamiento para Servicios de Hotelería · Hotel Booking Demand

Proyecto de ciencia de datos (Maestría en Inteligencia Artificial, Universidad de La Salle) desarrollado
con la metodología **CRISP-DM** sobre el dataset *Hotel Booking Demand*. El repositorio contiene:

1. **Módulos de entrenamiento** ejecutables desde la terminal (`python main.py`), con logs de cada etapa.
   Entrenan tres modelos y eligen los campeones con la estrategia SHAP + Pareto:
   - **Anticipación** de la reserva (`lead_time`, regresión).
   - **Propensión a cancelar** (`is_canceled`, clasificación).
   - **Segmentación** de reservas por comportamiento (clustering).
2. **Tablero en Dash** que cuenta el storytelling del proyecto y permite hacer **predicciones en tiempo
   real** con los modelos campeones. Se publica en **Railway**.

Autor: Diego Nicolás Ávila Moreno.

---

## Estructura

```
├── main.py                    # CLI del entrenamiento
├── config.yaml                # semilla, umbrales SHAP, tolerancias, parámetros
├── src/                       # código compartido por entrenamiento y tablero
│   ├── logging_config.py      # logging a terminal (con colores) y a archivo
│   ├── datos.py               # lectura y validación del CSV
│   ├── faltantes.py           # diagnóstico MCAR/MAR/MNAR e imputación
│   ├── features.py            # 31 variables derivadas + bases sin fuga
│   └── segmentos.py           # asignador de segmentos (usado por el tablero)
├── training/
│   ├── pipeline.py            # orquestador
│   ├── eda.py                 # agregados del storytelling
│   ├── regresion.py           # lead_time: 4 etapas + campeón
│   ├── clasificacion.py       # cancelación: 4 etapas + calibración + campeón
│   ├── segmentacion.py        # K-Means, GMM, Ward, Aglomerativo, DBSCAN + perfiles
│   ├── importancia.py         # SHAP (o permutación) + selección Pareto
│   └── artefactos.py          # exportación de modelos, tablas y metadatos
├── artifacts/                 # salida del entrenamiento (la lee el tablero)
├── dashboard/
│   ├── app.py                 # app Dash (server = app.server para gunicorn)
│   ├── servicios.py           # carga de artefactos y lógica de predicción (sin Dash)
│   ├── formulario.py          # formulario de reserva de los simuladores
│   ├── graficos.py, tema.py, componentes.py
│   ├── pages/                 # una página por fase del storytelling
│   └── assets/style.css       # estilos (paleta azul / amarillo de alto contraste)
├── tests/
├── data/raw/                  # aquí va hotel_bookings.csv (no se versiona)
├── requirements.txt           # dependencias del tablero (Railway)
├── requirements-train.txt     # dependencias del entrenamiento
└── Procfile                   # comando de arranque en Railway
```

## 1. Instalación

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
```

Descarga `hotel_bookings.csv` desde
[Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) y ubícalo en `data/raw/`.

## 2. Entrenamiento desde la terminal

```bash
python main.py                                    # todas las tareas (≈ 10-20 min)
python main.py --tareas cancelacion               # una sola tarea
python main.py --tareas eda segmentacion          # varias tareas
python main.py --rapido                           # prueba rápida (no publicar estos modelos)
python main.py --nivel-log DEBUG                  # más detalle en la terminal
python main.py --datos ruta/hotel_bookings.csv    # otra ubicación del CSV
python main.py --help
```

Tareas: `eda`, `lead_time`, `cancelacion`, `segmentacion` (o `todas`). Siempre se ejecuta antes el
preprocesamiento común: carga → diagnóstico de faltantes → imputación → variables derivadas.

Cada etapa registra su inicio, su fin y su duración; cada modelo registra sus métricas al terminar:

```
2026-11-14 11:56:00 | INFO     | training.clasificacion   | ▶ Inicia: Etapa 2/4 - Importancia SHAP y selección Pareto (cancelación)
2026-11-14 11:56:00 | INFO     | training.clasificacion   | Modelo base para SHAP: Random Forest (mejor modelo de árboles por AUC)
2026-11-14 11:56:16 | INFO     | training.importancia     | Método: Permutación (shap no instalado) | variables: 33 -> 14 seleccionadas (acumulan 90.6 % >= 90 %)
2026-11-14 11:56:16 | INFO     | training.importancia     |    country                           22.20 %  (acum.  22.2 %)
...
2026-11-14 11:56:16 | INFO     | training.clasificacion   | ✔ Termina: Etapa 2/4 - Importancia SHAP y selección Pareto (cancelación) (16.8 s)
```

(Ejemplo real de una ejecución sin `shap` ni `xgboost`: con ambas librerías instaladas el log indica
`SHAP - TreeExplainer` y XGBoost participa en la comparación.)

El log completo (nivel DEBUG, con las tablas detalladas) se guarda en `logs/entrenamiento_AAAAMMDD_HHMMSS.log`.

### Artefactos que se exportan

| Carpeta | Contenido |
|---|---|
| `artifacts/eda/` | indicadores, agregados por mes/hotel, tasas de cancelación por factor, faltantes, fuga, opciones de formularios |
| `artifacts/cancelacion/` | `campeon.joblib` (pipeline completo), `metadata.json`, comparación de modelos, importancia SHAP, muestra SHAP, curvas ROC, calibración, probabilidades de test |
| `artifacts/lead_time/` | `campeon.joblib`, `metadata.json`, comparación, importancia SHAP, predicciones de test |
| `artifacts/segmentacion/` | `asignador.joblib`, `metadata.json` (reglas del árbol, nombres), perfiles, z-scores, pruebas, PCA |

`metadata.json` guarda las variables del campeón, su tipo, rango y categorías, las métricas y las
versiones de las librerías. El tablero usa esa información para construir y marcar el formulario.

## 3. Tablero

```bash
pip install -r requirements.txt
python -m dashboard.app            # http://127.0.0.1:8050
```

Páginas (una por fase CRISP-DM):

| Página | Contenido |
|---|---|
| El problema | Indicadores, demanda y tarifa por mes y hotel, objetivos |
| Calidad de los datos | Faltantes y su tratamiento, duplicados, fuga de información |
| Qué explica la cancelación | Tasas por factor, asociación, antelación y tarifa por grupo, composición |
| Predicción de cancelación | Simulador, comparación de modelos, importancia SHAP (Pareto), ROC, calibración, umbral interactivo |
| Predicción de anticipación | Simulador, comparación, importancia SHAP, real vs. predicho |
| Segmentos de clientes | Perfiles, rasgos (z-scores), PCA, reglas, asignador de segmento |
| Recomendaciones | Usos, limitaciones y próximos pasos |

SHAP se calcula **solo durante el entrenamiento**; el tablero muestra la importancia como gráfica
y no necesita la librería `shap`, lo que reduce la memoria en Railway.

## 4. Publicación en Railway

1. Entrena con `python main.py` (con `xgboost` y `shap` instalados) y verifica con `pytest`.
2. Sube el repositorio a GitHub **incluyendo la carpeta `artifacts/`** (el CSV y los logs se ignoran).
3. En Railway: *New Project → Deploy from GitHub repo* y elige el repositorio.
4. Railway detecta Python por `requirements.txt` y arranca con el `Procfile`:
   `gunicorn dashboard.app:server --bind 0.0.0.0:$PORT`. No se requieren variables de entorno.
5. En *Settings → Networking* genera el dominio público.

Recomendaciones:

- **Versiones:** `scikit-learn`, `xgboost`, `pandas` y `numpy` de `requirements.txt` deben coincidir con
  las del entrenamiento (ver `"versiones"` en `metadata.json`). Al arrancar, el tablero imprime un aviso
  si no coinciden.
- **Tamaño de los modelos:** con XGBoost como campeón los modelos pesan pocos MB. Si el campeón resulta
  ser Random Forest (por ejemplo, si no está instalado `xgboost`), cada modelo pesa ≈ 60 MB y consume más
  memoria; GitHub rechaza archivos de más de 100 MB (el log avisa si se supera).

## 5. Pruebas

```bash
pytest -q
```

Verifican las variables derivadas, la exclusión de variables con fuga y, si existen artefactos, que
los modelos carguen y predigan valores válidos.

## Notas sobre la adaptación del notebook

- Los `print`/`display` del notebook se reemplazaron por logging y los gráficos de matplotlib por
  artefactos que grafica el tablero.
- El campeón de cancelación se elige entre los modelos con probabilidad (se excluye SVM, que solo entrega
  un puntaje), porque el tablero y las decisiones de negocio usan la probabilidad.
- Los parámetros que no eran legibles en la versión impresa del notebook se fijaron así:
  `min_samples_leaf=3` (Random Forest de regresión) y `min_samples_leaf=2` (Random Forest de clasificación).
- La imputación MICE usa solo predictores numéricos; en este dataset no se activa (country se imputa con
  clasificador y children con la moda).

Datos: António, N., de Almeida, A. y Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22,
41-49. Licencia CC BY 4.0.
