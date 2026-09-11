"""
Interfaz de Usuario (Streamlit) — Predicción de Nota Final G3 (Dataset UCI student-mat.csv, escala 0-20)
Refactor: Adaptado a la escala 0-20 y predicción temprana basada en G1 y comportamiento/demografía.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de Nota Final G3",
    page_icon="🎯",
    layout="centered",
)

def _html(*parts: str) -> str:
    return "".join(parts)

# ---------------------------------------------------------------------------
# Identidad visual
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #F3F5FA; }
h1, h2, h3 { font-family: 'Lora', serif !important; color: #1E2A44 !important; }

section[data-testid="stSidebar"] {
    background-color: #E7EBF3;
    border-right: 1px solid rgba(30,42,68,0.12);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 1.05rem !important;
    color: #4B5568 !important;
    margin-top: 1.4rem;
}

.app-header { padding: 0.2rem 0 1.2rem 0; border-bottom: 1px solid rgba(30,42,68,0.12); margin-bottom: 1.6rem; }
.app-header .eyebrow { font-size: 0.85rem; color: #8C2F39; font-weight: 600; }
.app-header h1 { font-size: 1.9rem; margin: 0.15rem 0 0.3rem 0; }
.app-header p { color: #5B6577; font-size: 0.95rem; margin: 0; max-width: 48ch; }

.card {
    background: #FFFFFF;
    border: 1px solid rgba(30,42,68,0.10);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.3rem;
}

/* --- Tacómetro (gauge semicircular con CSS conic-gradient, escala 0-20) --- */
.gauge-wrap { position: relative; width: 260px; height: 140px; margin: 0.4rem auto 0.2rem auto; overflow: hidden; }
.gauge-arc {
    position: absolute; top: 0; left: 0; width: 260px; height: 260px; border-radius: 50%;
    background: conic-gradient(from 270deg,
        #8C2F39 0deg 90deg,
        #B85C2E 90deg 108deg,
        #6E8F4C 108deg 144deg,
        #2E6B4F 144deg 180deg,
        transparent 180deg 360deg
    );
}
.gauge-hole {
    position: absolute; left: 40px; top: 40px; width: 180px; height: 180px;
    background: #FFFFFF; border-radius: 50%;
}
.gauge-needle {
    position: absolute; left: 50%; bottom: 0; width: 4px; height: 108px;
    background: #1E2A44; transform-origin: bottom center; border-radius: 2px 2px 0 0;
    transition: transform 0.25s ease-out;
}
.gauge-pivot {
    position: absolute; left: 50%; bottom: -7px; width: 16px; height: 16px;
    background: #1E2A44; border-radius: 50%; transform: translateX(-50%);
}
.gauge-scale { display: flex; justify-content: space-between; width: 260px; margin: 0 auto;
    font-size: 0.78rem; color: #5B6577; }

.kpi-number { text-align: center; font-family: 'Lora', serif; font-weight: 700; color: #1E2A44;
    font-size: 2.6rem; margin-top: 0.3rem; }
.kpi-band { text-align: center; font-size: 1rem; font-weight: 600; margin-top: -0.3rem; }

.limit-note {
    font-size: 0.85rem; color: #5B6577; border-left: 3px solid #1E2A44;
    padding: 0.6rem 0.9rem; background: #E7EBF3; border-radius: 0 6px 6px 0;
}
.sidebar-rule { border-top: 1px solid rgba(30,42,68,0.15); margin: 0.6rem 0 0.2rem 0; }
.feature-note { font-size: 0.82rem; color: #5B6577; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bandas de score para escala 0 a 20
# ---------------------------------------------------------------------------
def banda_score(valor: float):
    if valor < 10:
        return "#8C2F39", "Reprobado"
    elif valor < 12:
        return "#B85C2E", "Suficiente"
    elif valor < 16:
        return "#6E8F4C", "Bueno"
    return "#2E6B4F", "Sobresaliente"


def render_gauge(valor: float) -> str:
    """Genera el HTML/CSS del tacómetro semicircular para un valor de 0 a 20."""
    valor_clip = max(0.0, min(20.0, valor))
    angulo = (valor_clip / 20.0) * 180.0 - 90.0  # -90° (izq, 0) .. +90° (der, 20)
    color, _ = banda_score(valor_clip)
    return _html(
        '<div class="gauge-wrap">',
        '<div class="gauge-arc"></div>',
        '<div class="gauge-hole"></div>',
        f'<div class="gauge-needle" style="transform: translateX(-50%) rotate({angulo:.1f}deg); background:{color};"></div>',
        '<div class="gauge-pivot"></div>',
        '</div>',
        '<div class="gauge-scale"><span>0</span><span>10</span><span>20</span></div>',
    )

# ---------------------------------------------------------------------------
# Carga del modelo y artefactos
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    model = joblib.load("total_score_model.pkl")
    scaler = joblib.load("total_score_scaler.pkl")
    poly = joblib.load("total_score_poly.pkl")
    feature_columns = joblib.load("total_score_features.pkl")
    mae_test = joblib.load("total_score_mae.pkl")
    return model, scaler, poly, feature_columns, mae_test

try:
    model, scaler, poly, feature_columns, mae_test = cargar_artefactos()
except FileNotFoundError:
    st.error(
        "No se encontraron los artefactos del modelo (total_score_model.pkl, "
        "total_score_scaler.pkl, total_score_poly.pkl, total_score_features.pkl, "
        "total_score_mae.pkl). Ejecuta primero 'train_total_score_model.py'."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    _html(
        '<div class="app-header">',
        '<div class="eyebrow">Herramienta de apoyo docente — Dataset UCI</div>',
        '<h1>Predicción de Nota Final G3</h1>',
        '<p>Estimación temprana de la nota final (0–20 puntos) a partir de la nota parcial G1 '
        'y el perfil de comportamiento del estudiante.</p>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panel lateral — Controles según las features seleccionadas (|r| >= 0.10)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Perfil del estudiante")
st.sidebar.markdown(
    _html(
        '<p class="feature-note">Ingresa los datos del estudiante para calcular la nota final predicha (G3) sobre 20 puntos.</p>',
    ),
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Rendimiento y hábitos")
g1 = st.sidebar.slider("Nota 1er Parcial (G1)", 0.0, 20.0, 12.0, help="Nota obtenida en el primer periodo (0-20)")
failures = st.sidebar.slider("Materias reprobadas previas", 0, 3, 0)
studytime_opt = st.sidebar.selectbox(
    "Tiempo de estudio semanal",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "1: <2 horas", 2: "2: 2 a 5 horas", 3: "3: 5 a 10 horas", 4: "4: >10 horas"}[x],
    index=1
)
goout_opt = st.sidebar.selectbox(
    "Frecuencia de salidas con amigos",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: f"{x}: {'Muy poco' if x==1 else 'Mucho' if x==5 else 'Medio'}",
    index=2
)

st.sidebar.markdown(_html('<div class="sidebar-rule"></div>'), unsafe_allow_html=True)
st.sidebar.markdown("### Contexto personal y familiar")
age = st.sidebar.slider("Edad", 15, 22, 17)
traveltime_opt = st.sidebar.selectbox(
    "Tiempo de traslado a la escuela",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1: "1: <15 min", 2: "2: 15-30 min", 3: "3: 30-60 min", 4: "4: >1 hora"}[x],
    index=0
)
medu_opt = st.sidebar.selectbox(
    "Educación de la madre (Medu)",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {0: "0: Ninguna", 1: "1: Primaria", 2: "2: 5º-9º grado", 3: "3: Secundaria", 4: "4: Superior"}[x],
    index=2
)
fedu_opt = st.sidebar.selectbox(
    "Educación del padre (Fedu)",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: {0: "0: Ninguna", 1: "1: Primaria", 2: "2: 5º-9º grado", 3: "3: Secundaria", 4: "4: Superior"}[x],
    index=2
)
mjob = st.sidebar.selectbox("Trabajo de la madre", ["health", "services", "at_home", "teacher", "other"])

st.sidebar.markdown(_html('<div class="sidebar-rule"></div>'), unsafe_allow_html=True)
st.sidebar.markdown("### Factores socioeducativos")
higher = st.sidebar.selectbox("¿Desea cursar educación superior?", ["yes", "no"])
internet = st.sidebar.selectbox("¿Acceso a internet en casa?", ["yes", "no"])
romantic = st.sidebar.selectbox("¿En relación amorosa?", ["no", "yes"])
paid = st.sidebar.selectbox("¿Clases particulares pagadas?", ["no", "yes"])

# ---------------------------------------------------------------------------
# Construcción del vector de features
# ---------------------------------------------------------------------------
fila = {col: 0 for col in feature_columns}

if "G1" in fila: fila["G1"] = g1
if "failures" in fila: fila["failures"] = failures
if "studytime" in fila: fila["studytime"] = studytime_opt
if "goout" in fila: fila["goout"] = goout_opt
if "traveltime" in fila: fila["traveltime"] = traveltime_opt
if "age" in fila: fila["age"] = age
if "Medu" in fila: fila["Medu"] = medu_opt
if "Fedu" in fila: fila["Fedu"] = fedu_opt

for col_prefix, val in [
    ("higher", higher), ("paid", paid), ("internet", internet),
    ("romantic", romantic), ("Mjob", mjob)
]:
    dummy_col = f"{col_prefix}_{val}"
    if dummy_col in fila:
        fila[dummy_col] = 1

perfil_df = pd.DataFrame([fila])[feature_columns]

# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
perfil_scaled = scaler.transform(perfil_df)
if poly is not None:
    perfil_scaled_df = pd.DataFrame(perfil_scaled, columns=feature_columns)
    X_final = poly.transform(perfil_scaled_df)
else:
    X_final = perfil_scaled

prediccion = float(model.predict(X_final)[0])
prediccion_clip = max(0.0, min(20.0, prediccion))

color_banda, etiqueta_banda = banda_score(prediccion_clip)

# --- Tarjeta: Tacómetro + KPI -----------------------------------------------
st.markdown(
    _html(
        '<div class="card">',
        render_gauge(prediccion_clip),
        f'<div class="kpi-number">{prediccion_clip:.1f}<span style="font-size:1.1rem;color:#5B6577;"> / 20</span></div>',
        f'<div class="kpi-band" style="color:{color_banda};">{etiqueta_banda}</div>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# --- Tarjeta: Interpretación del error ---------------------------------------
st.markdown(
    _html(
        '<div class="card">',
        '<p style="margin:0; color:#1E2A44; font-size:1rem;">',
        f'El modelo predice una nota final (G3) de <b>{prediccion_clip:.1f} / 20</b>. ',
        'Basado en la validación del set de prueba (UCI), la estimación tiene un margen de error promedio de ',
        f'<b>+/- {mae_test:.2f} puntos</b> (MAE sobre el conjunto de test).',
        '</p>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# --- Detalle del perfil ingresado -------------------------------------------
with st.expander("Ver el perfil completo ingresado"):
    resumen = pd.DataFrame({
        "Variable": ["Nota 1er Parcial (G1)", "Materias Reprobadas", "Tiempo de Estudio", "Edad", "Educación Madre (Medu)", "Desea Superior", "Internet en Casa"],
        "Valor": [str(v) for v in [g1, failures, f"Categoría {studytime_opt}", age, medu_opt, higher, internet]],
    })
    st.dataframe(resumen, hide_index=True, width="stretch")

# --- Nota metodológica ------------------------------------------------------
st.markdown(
    _html(
        '<div class="limit-note">',
        '<b>Dataset UCI Student Performance:</b> Este modelo predice la nota final (G3, escala 0-20) ',
        'utilizando la nota parcial G1 y factores socio-conductuales. Al excluir las notas intermedias G2, ',
        'se logra una predicción temprana genuina para la detección a tiempo de estudiantes en riesgo.',
        '</div>',
    ),
    unsafe_allow_html=True,
)