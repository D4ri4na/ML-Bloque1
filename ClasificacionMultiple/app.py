"""
Interfaz de Usuario (Streamlit) — Predicción de Total_Score (regresión continua)
Refactor: de clasificación de Grade -> regresión de Total_Score.

Requisitos previos: ejecutar primero el notebook 'Proyecto_Regresion_TotalScore.ipynb'
completo (genera total_score_model.pkl, total_score_scaler.pkl, total_score_poly.pkl,
total_score_features.pkl y total_score_mae.pkl en la misma carpeta).

Ejecutar con:
    pip install streamlit joblib scikit-learn pandas numpy
    streamlit run app.py

Para el theming nativo (colores de sliders/botones), copia streamlit_config.toml a
.streamlit/config.toml dentro de la carpeta del proyecto.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predicción de Total Score",
    page_icon="🎯",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helper: concatena fragmentos HTML en una sola línea, SIN saltos de línea
# ni espacios de indentación. Esto es crítico: Streamlit pasa unsafe_allow_html
# por el parser de Markdown antes de renderizar, y cualquier línea en blanco o
# con sangría dentro del HTML hace que Markdown cierre el bloque HTML crudo y
# empiece a tratar el resto como un "code block" (de ahí que se vieran las
# etiquetas <div> como texto literal en vez de renderizarse).
# ---------------------------------------------------------------------------
def _html(*parts: str) -> str:
    return "".join(parts)

# ---------------------------------------------------------------------------
# Identidad visual (misma paleta "boletín académico" del proyecto de clasificación)
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

/* --- Tacómetro (gauge semicircular con CSS conic-gradient) ---
   OJO: el gradiente debe arrancar en 270deg (izquierda, 9 en punto), no en
   180deg (abajo, 6 en punto). Con 180deg el semicírculo de color quedaba
   rotado 90° y su mitad se pintaba por DEBAJO del centro, desbordando el
   contenedor de 140px y tapando el número/banda que van justo después. */
.gauge-wrap { position: relative; width: 260px; height: 140px; margin: 0.4rem auto 0.2rem auto; overflow: hidden; }
.gauge-arc {
    position: absolute; top: 0; left: 0; width: 260px; height: 260px; border-radius: 50%;
    background: conic-gradient(from 270deg,
        #8C2F39 0deg 54deg,
        #B85C2E 54deg 90deg,
        #B07C1D 90deg 126deg,
        #6E8F4C 126deg 153deg,
        #2E6B4F 153deg 180deg,
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
    font-size: 0.85rem; color: #5B6577; border-left: 3px solid #8C2F39;
    padding: 0.6rem 0.9rem; background: #F5E6E4; border-radius: 0 6px 6px 0;
}
.sidebar-rule { border-top: 1px solid rgba(30,42,68,0.15); margin: 0.6rem 0 0.2rem 0; }
.feature-note { font-size: 0.82rem; color: #5B6577; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Bandas de score -> color y etiqueta (el burdeos queda reservado para el
# score más bajo, igual que en el proyecto de clasificación)
# ---------------------------------------------------------------------------
def banda_score(valor: float):
    if valor < 30:
        return "#8C2F39", "Crítico"
    elif valor < 50:
        return "#B85C2E", "Bajo"
    elif valor < 70:
        return "#B07C1D", "Medio"
    elif valor < 85:
        return "#6E8F4C", "Bueno"
    return "#2E6B4F", "Sobresaliente"


def render_gauge(valor: float) -> str:
    """Genera el HTML/CSS del tacómetro semicircular para un valor 0-100.
    Devuelve todo en una sola línea (sin \\n ni indentación) a propósito."""
    valor_clip = max(0.0, min(100.0, valor))
    angulo = (valor_clip / 100.0) * 180.0 - 90.0  # -90° (izq, 0) .. +90° (der, 100)
    color, _ = banda_score(valor_clip)
    return _html(
        '<div class="gauge-wrap">',
        '<div class="gauge-arc"></div>',
        '<div class="gauge-hole"></div>',
        f'<div class="gauge-needle" style="transform: translateX(-50%) rotate({angulo:.1f}deg); background:{color};"></div>',
        '<div class="gauge-pivot"></div>',
        '</div>',
        '<div class="gauge-scale"><span>0</span><span>50</span><span>100</span></div>',
    )

# ---------------------------------------------------------------------------
# Carga del modelo, escalador, transformador polinomial (si aplica), features y MAE
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    model = joblib.load("total_score_model.pkl")
    scaler = joblib.load("total_score_scaler.pkl")
    poly = joblib.load("total_score_poly.pkl")          # None si el mejor modelo fue el lineal
    feature_columns = joblib.load("total_score_features.pkl")
    mae_test = joblib.load("total_score_mae.pkl")
    return model, scaler, poly, feature_columns, mae_test

try:
    model, scaler, poly, feature_columns, mae_test = cargar_artefactos()
except FileNotFoundError:
    st.error(
        "No se encontraron los artefactos del modelo (total_score_model.pkl, "
        "total_score_scaler.pkl, total_score_poly.pkl, total_score_features.pkl, "
        "total_score_mae.pkl). Ejecuta primero todas las celdas del notebook "
        "'Proyecto_Regresion_TotalScore.ipynb' y colócalos en esta misma carpeta."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------
st.markdown(
    _html(
        '<div class="app-header">',
        '<div class="eyebrow">Herramienta de apoyo docente</div>',
        '<h1>Predicción de Total Score</h1>',
        '<p>Estima el puntaje total (0–100) de un estudiante a partir de su '
        'comportamiento y contexto demográfico — sin usar notas parciales.</p>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panel lateral — únicamente las variables que sobrevivieron a la selección
# por correlación en el notebook (ver Sección 2.5 del pipeline de entrenamiento)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Perfil del estudiante")
st.sidebar.markdown(
    _html(
        '<p class="feature-note">Estos son los únicos campos que el modelo usa: '
        'son las variables que, tras el filtro de correlación, quedaron seleccionadas '
        'en el notebook de entrenamiento.</p>',
    ),
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Comportamiento y asistencia")
attendance = st.sidebar.slider("Asistencia (%)", 0.0, 100.0, 80.0)
age = st.sidebar.slider("Edad", 17, 30, 20)

st.sidebar.markdown(_html('<div class="sidebar-rule"></div>'), unsafe_allow_html=True)
st.sidebar.markdown("### Contexto")
internet = st.sidebar.selectbox("Acceso a internet en casa", ["Yes", "No"])
department = st.sidebar.selectbox("Departamento", ["CS", "Engineering", "Business", "Mathematics"])
income = st.sidebar.selectbox("Nivel de ingreso familiar", ["Medium", "Low", "High"])
parent_edu = st.sidebar.selectbox(
    "Nivel educativo de los padres", ["Unknown", "High School", "Bachelor's", "Master's", "PhD"]
)

st.sidebar.markdown(_html('<div class="sidebar-rule"></div>'), unsafe_allow_html=True)
escalar_escala = st.sidebar.checkbox(
    "Escalar salida a todo el rango (0 – 100)",
    value=True,
    help="Mapea la predicción relativa del modelo para ocupar todo el espectro visual de 0 a 100."
)
st.sidebar.caption(
    "Nota: dentro de Departamento, Ingreso e Educación de los padres, el modelo solo "
    "distingue 'CS', 'Medium' y 'Unknown' respectivamente frente a todo lo demás — "
    "son las únicas categorías que superaron el filtro de correlación."
)

# ---------------------------------------------------------------------------
# Construcción del vector de features EXACTAMENTE con las columnas de entrenamiento
# ---------------------------------------------------------------------------
fila = {col: 0 for col in feature_columns}

if "Attendance (%)" in fila:
    fila["Attendance (%)"] = attendance
if "Age" in fila:
    fila["Age"] = age
if "Internet_Access_at_Home" in fila:
    fila["Internet_Access_at_Home"] = 1 if internet == "Yes" else 0

# Variables dummy: se activa la columna correspondiente solo si existe entre las
# features seleccionadas (la categoría base o "el resto" queda en 0)
for col_name, valor in [
    ("Department", department), ("Family_Income_Level", income), ("Parent_Education_Level", parent_edu),
]:
    dummy_col = f"{col_name}_{valor}"
    if dummy_col in fila:
        fila[dummy_col] = 1

perfil_df = pd.DataFrame([fila])[feature_columns]

# ---------------------------------------------------------------------------
# Predicción en tiempo real
# ---------------------------------------------------------------------------
perfil_scaled = scaler.transform(perfil_df)
if poly is not None:
    perfil_scaled_df = pd.DataFrame(perfil_scaled, columns=feature_columns)
    X_final = poly.transform(perfil_scaled_df)
else:
    X_final = perfil_scaled

prediccion_raw = float(model.predict(X_final)[0])

# Límites del modelo para el espacio de entrada de las features
PRED_MIN = 73.5810
PRED_MAX = 79.4278

if escalar_escala:
    # Transformación Min-Max -> Rango 0 a 100
    prediccion = ((prediccion_raw - PRED_MIN) / (PRED_MAX - PRED_MIN)) * 100.0
else:
    prediccion = prediccion_raw

prediccion_clip = max(0.0, min(100.0, prediccion))

color_banda, etiqueta_banda = banda_score(prediccion_clip)

# --- Tarjeta: tacómetro + KPI -----------------------------------------------
# TODO el card (arco + aguja + número + banda) se arma como UNA sola cadena
# sin saltos de línea. Esto es lo que antes fallaba: al estar repartido en un
# f"""...""" multilínea con indentación, Markdown cerraba el <div> a mitad de
# camino y el resto se mostraba como texto/código plano.
st.markdown(
    _html(
        '<div class="card">',
        render_gauge(prediccion_clip),
        f'<div class="kpi-number">{prediccion:.1f}<span style="font-size:1.1rem;color:#5B6577;"> / 100</span></div>',
        f'<div class="kpi-band" style="color:{color_banda};">{etiqueta_banda}</div>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# --- Tarjeta: interpretación del error ---------------------------------------
st.markdown(
    _html(
        '<div class="card">',
        '<p style="margin:0; color:#1E2A44; font-size:1rem;">',
        f'El modelo predice un puntaje de <b>{prediccion:.1f}</b>. ',
        'Basado en el entrenamiento, esta estimación tiene un margen de error ',
        f'promedio de <b>+/- {mae_test:.1f} puntos</b> (MAE sobre el set de prueba).',
        '</p>',
        '</div>',
    ),
    unsafe_allow_html=True,
)

# --- Detalle del perfil ingresado (opcional, plegado) -----------------------
with st.expander("Ver el perfil ingresado"):
    resumen = pd.DataFrame({
        "Variable": ["Asistencia (%)", "Edad", "Internet en casa", "Departamento",
                     "Ingreso familiar", "Educación de los padres"],
        "Valor": [str(v) for v in [attendance, age, internet, department, income, parent_edu]],
    })
    st.dataframe(resumen, hide_index=True, width="stretch")

# --- Nota de limitación del modelo ------------------------------------------
st.markdown(
    _html(
        '<div class="limit-note">',
        'Este modelo se entrenó sobre el dataset "Biased", donde ninguna variable de ',
        'comportamiento o demográfica alcanzó una correlación relevante (|r| ≥ 0.05) con ',
        'Total_Score — el R² de test es cercano a 0. En la práctica, el modelo se comporta ',
        'como una estimación cercana al promedio histórico ajustada por señales muy débiles. ',
        'Usa esta predicción únicamente como referencia orientativa, no como una medida ',
        'precisa del desempeño del estudiante.',
        '</div>',
    ),
    unsafe_allow_html=True,
)