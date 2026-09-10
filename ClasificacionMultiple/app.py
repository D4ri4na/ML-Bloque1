"""
Interfaz de Usuario (Streamlit) — Predicción de Grade del estudiante
Paso 3.5 de la Rúbrica: Despliegue de la Interfaz de Usuario (UI)

Requisitos previos:
- Ejecutar 'Proyecto_Clasificacion_Multiple.ipynb' o 'train_model.py'
  (genera grade_model.pkl, grade_scaler.pkl y feature_columns.pkl).

Ejecución:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración inicial de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Rendimiento Académico",
    page_icon="🎓",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Identidad visual y paleta accesible "boletín académico"
# ---------------------------------------------------------------------------
# Se ajustó 'C' a #946000 para cumplir con contraste accesible WCAG AA (texto blanco)
GRADE_COLORS = {
    "A": "#2E6B4F",  # Verde bosque
    "B": "#5A7D36",  # Verde oliva
    "C": "#946000",  # Ámbar oscuro accesible
    "D": "#B85C2E",  # Naranja terracota
    "F": "#8C2F39",  # Burdeos de riesgo
}

GRADE_MESSAGES = {
    "A": "Desempeño sobresaliente. Sin señales de riesgo.",
    "B": "Desempeño sólido, sin alertas relevantes.",
    "C": "Desempeño en observación: conviene dar seguimiento cercano.",
    "D": "Riesgo de reprobación. Se recomienda asignar tutoría.",
    "F": "Riesgo alto de reprobación. Se recomienda tutoría prioritaria.",
}

CSS_THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

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

/* Encabezado principal */
.app-header { padding: 0.2rem 0 1.2rem 0; border-bottom: 1px solid rgba(30,42,68,0.12); margin-bottom: 1.6rem; }
.app-header .eyebrow { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #8C2F39; font-weight: 600; }
.app-header h1 { font-size: 1.9rem; margin: 0.15rem 0 0.3rem 0; }
.app-header p { color: #5B6577; font-size: 0.95rem; margin: 0; max-width: 46ch; }

/* Tarjetas contenedoras */
.card {
    background: #FFFFFF;
    border: 1px solid rgba(30,42,68,0.10);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.3rem;
}

/* Insignia de predicción */
.badge-row { display: flex; align-items: center; gap: 1.3rem; }
.badge {
    width: 88px; height: 88px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Lora', serif; font-weight: 700; font-size: 2.3rem; color: #FFFFFF;
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(30,42,68,0.18);
}
.badge-text .label { font-size: 0.85rem; color: #5B6577; margin-bottom: 0.15rem; }
.badge-text .msg { font-size: 1.02rem; color: #1E2A44; font-weight: 500; max-width: 40ch; }

/* Barras de probabilidad */
.prob-row { display: flex; align-items: center; gap: 0.7rem; margin: 0.55rem 0; }
.prob-chip {
    width: 30px; height: 30px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    color: #FFFFFF; font-weight: 600; font-size: 0.92rem; font-family: 'Lora', serif;
}
.prob-track { flex-grow: 1; background: #EDF0F6; border-radius: 5px; height: 14px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 5px; }
.prob-pct { width: 52px; text-align: right; font-size: 0.88rem; color: #4B5568; font-variant-numeric: tabular-nums; }

/* Caja de limitación */
.limit-note {
    font-size: 0.85rem; color: #5B6577; border-left: 3px solid #8C2F39;
    padding: 0.6rem 0.9rem; background: #F5E6E4; border-radius: 0 6px 6px 0;
}

/* Separadores de sidebar */
.sidebar-rule { border-top: 1px solid rgba(30,42,68,0.15); margin: 0.6rem 0 0.2rem 0; }
</style>
"""
st.markdown(CSS_THEME, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Carga de artefactos del modelo
# ---------------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    model = joblib.load("grade_model.pkl")
    scaler = joblib.load("grade_scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = cargar_artefactos()
except FileNotFoundError:
    st.error(
        "No se encontraron los artefactos 'grade_model.pkl', 'grade_scaler.pkl' o 'feature_columns.pkl'. "
        "Ejecuta primero el script 'train_model.py' o el notebook 'Proyecto_Clasificacion_Multiple.ipynb'."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Diccionarios de traducción (Español -> Dataset original)
# ---------------------------------------------------------------------------
MAP_GENDER = {"Femenino": "Female", "Masculino": "Male"}
MAP_DEPT = {
    "Ingeniería": "Engineering",
    "Administración / Negocios": "Business",
    "Matemáticas": "Mathematics",
    "Ciencias de la Computación": "CS",
}
MAP_PARENT_EDU = {
    "Desconocido": "Unknown",
    "Secundaria / Bachillerato": "High School",
    "Licenciatura / Pregrado": "Bachelor's",
    "Maestría": "Master's",
    "Doctorado (PhD)": "PhD",
}
MAP_INCOME = {"Bajo": "Low", "Medio": "Medium", "Alto": "High"}

# ---------------------------------------------------------------------------
# Manejo del estado del Total Score (st.session_state)
# ---------------------------------------------------------------------------
def recalcular_total():
    """Recalcula el Total_Score como el promedio de las notas individuales."""
    notas = [
        st.session_state.midterm,
        st.session_state.final,
        st.session_state.assignments,
        st.session_state.quizzes,
        st.session_state.projects,
    ]
    st.session_state.total_score = round(float(np.mean(notas)), 2)

# Inicializar st.session_state si no existe
if "total_score" not in st.session_state:
    st.session_state.total_score = 70.0

# ---------------------------------------------------------------------------
# Encabezado principal
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>Predicción de rendimiento académico</h1>
        <p>Ingresa el perfil de un estudiante para estimar su categoría de
        desempeño (Grade) y decidir si requiere tutoría preventiva.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panel lateral (Inputs del usuario)
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Perfil del estudiante")

st.sidebar.markdown("### Datos generales")
gender_es = st.sidebar.selectbox("Género", list(MAP_GENDER.keys()))
dept_es = st.sidebar.selectbox("Departamento", list(MAP_DEPT.keys()))
age = st.sidebar.slider("Edad", 17, 30, 20)

st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### Desempeño académico")
attendance = st.sidebar.slider("Asistencia (%)", 0.0, 100.0, 85.0)
midterm = st.sidebar.slider("Nota parcial", 0.0, 100.0, 70.0, key="midterm", on_change=recalcular_total)
final = st.sidebar.slider("Nota final", 0.0, 100.0, 70.0, key="final", on_change=recalcular_total)
assignments = st.sidebar.slider("Promedio de tareas", 0.0, 100.0, 70.0, key="assignments", on_change=recalcular_total)
quizzes = st.sidebar.slider("Promedio de quizzes", 0.0, 100.0, 70.0, key="quizzes", on_change=recalcular_total)
participation = st.sidebar.slider("Puntaje de participación", 0.0, 10.0, 5.0)
projects = st.sidebar.slider("Puntaje de proyectos", 0.0, 100.0, 70.0, key="projects", on_change=recalcular_total)

# Total score con session_state preservado
total_score = st.sidebar.slider(
    "Puntaje total (calculado, ajustable)", 0.0, 100.0, key="total_score"
)

st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### Contexto y bienestar")
study_hours = st.sidebar.slider("Horas de estudio / semana", 0.0, 40.0, 12.0)
sleep_hours = st.sidebar.slider("Horas de sueño / noche", 0.0, 12.0, 7.0)
stress = st.sidebar.slider("Nivel de estrés (1-10)", 1, 10, 5)
extracurricular_es = st.sidebar.selectbox("Actividades extracurriculares", ["No", "Sí"])
internet_es = st.sidebar.selectbox("Acceso a internet en casa", ["Sí", "No"])

st.sidebar.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
st.sidebar.markdown("### Entorno familiar")
parent_edu_es = st.sidebar.selectbox("Nivel educativo de los padres", list(MAP_PARENT_EDU.keys()))
income_es = st.sidebar.selectbox("Nivel de ingreso familiar", list(MAP_INCOME.keys()))

# ---------------------------------------------------------------------------
# Mapeo a formato del modelo y vector de características
# ---------------------------------------------------------------------------
gender = MAP_GENDER[gender_es]
department = MAP_DEPT[dept_es]
parent_edu = MAP_PARENT_EDU[parent_edu_es]
income = MAP_INCOME[income_es]
extracurricular = 1 if extracurricular_es == "Sí" else 0
internet = 1 if internet_es == "Sí" else 0

fila = {col: 0 for col in feature_columns}
fila["Age"] = age
fila["Attendance (%)"] = attendance
fila["Midterm_Score"] = midterm
fila["Final_Score"] = final
fila["Assignments_Avg"] = assignments
fila["Quizzes_Avg"] = quizzes
fila["Participation_Score"] = participation
fila["Projects_Score"] = projects
fila["Total_Score"] = total_score
fila["Study_Hours_per_Week"] = study_hours
fila["Sleep_Hours_per_Night"] = sleep_hours
fila["Stress_Level (1-10)"] = stress
fila["Extracurricular_Activities"] = extracurricular
fila["Internet_Access_at_Home"] = internet

for col_name, valor in [
    ("Gender", gender),
    ("Department", department),
    ("Parent_Education_Level", parent_edu),
    ("Family_Income_Level", income),
]:
    dummy_col = f"{col_name}_{valor}"
    if dummy_col in fila:
        fila[dummy_col] = 1

perfil_df = pd.DataFrame([fila])[feature_columns]

# ---------------------------------------------------------------------------
# Inferencias del modelo
# ---------------------------------------------------------------------------
perfil_scaled = scaler.transform(perfil_df)
pred_clase = model.predict(perfil_scaled)[0]
pred_proba = model.predict_proba(perfil_scaled)[0]

proba_df = (
    pd.DataFrame({"Clase": model.classes_, "Probabilidad": pred_proba})
    .sort_values("Probabilidad", ascending=False)
    .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Renderizado de resultados
# ---------------------------------------------------------------------------
badge_color = GRADE_COLORS.get(pred_clase, "#1E2A44")
mensaje = GRADE_MESSAGES.get(pred_clase, "")

# Tarjeta 1: Insignia principal
st.markdown(
    f"""
    <div class="card">
        <div class="badge-row">
            <div class="badge" style="background:{badge_color};">{pred_clase}</div>
            <div class="badge-text">
                <div class="label">Categoría de rendimiento predicha</div>
                <div class="msg">{mensaje}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tarjeta 2: Barras de probabilidad
filas_html = ""
for _, row in proba_df.iterrows():
    clase, prob = row["Clase"], row["Probabilidad"]
    color = GRADE_COLORS.get(clase, "#1E2A44")
    pct = prob * 100
    filas_html += f"""
    <div class="prob-row">
        <div class="prob-chip" style="background:{color};">{clase}</div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div>
        </div>
        <div class="prob-pct">{pct:.1f}%</div>
    </div>
    """

st.markdown(
    f"""
    <div class="card">
        <div style="font-family:'Lora',serif; font-weight:600; color:#1E2A44; margin-bottom:0.6rem;">
            Probabilidad por categoría
        </div>
        {filas_html}
    </div>
    """,
    unsafe_allow_html=True,
)

# Sección desplegable de perfil
with st.expander("Ver el perfil ingresado"):
    resumen = pd.DataFrame({
        "Variable": [
            "Género", "Departamento", "Edad", "Asistencia (%)", "Nota parcial", "Nota final",
            "Tareas (prom.)", "Quizzes (prom.)", "Participación", "Proyectos", "Puntaje total",
            "Horas de estudio/semana", "Horas de sueño/noche", "Nivel de estrés",
            "Extracurriculares", "Internet en casa", "Educación de los padres", "Ingreso familiar",
        ],
        "Valor": [
            str(v) for v in [
                gender_es, dept_es, age, attendance, midterm, final, assignments, quizzes,
                participation, projects, total_score, study_hours, sleep_hours, stress,
                extracurricular_es, internet_es, parent_edu_es, income_es,
            ]
        ],
    })
    st.dataframe(resumen, hide_index=True, use_container_width=True)

# Nota de limitación
st.markdown(
    """
    <div class="limit-note">
    Este modelo fue entrenado sobre el dataset "Biased", donde el Grade asignado
    tiene una relación débil con el desempeño académico medido (F1-macro ≈ 0.33
    en test). Usa esta predicción como apoyo, no como criterio único, y
    consulta la Sección 5 del informe para el detalle de esta limitación.
    </div>
    """,
    unsafe_allow_html=True,
)