"""
Script para entrenar el modelo de Regresión de Total_Score y generar los artefactos .pkl
necesarios para la interfaz Streamlit (app.py refactorizado).
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42

if __name__ == "__main__":
    print("1. Cargando dataset 'Students_Grading_Dataset_Biased.csv'...", flush=True)
    df = pd.read_csv("Students_Grading_Dataset_Biased.csv")
    print(f"   Dimensiones originales: {df.shape}", flush=True)

    cols_prohibidas = [
        "Grade", "Midterm_Score", "Final_Score", "Assignments_Avg",
        "Quizzes_Avg", "Participation_Score", "Projects_Score"
    ]
    cols_identificador = ["Student_ID", "First_Name", "Last_Name", "Email"]
    TARGET = "Total_Score"

    df_reg = df.drop(columns=cols_prohibidas + cols_identificador)

    print("2. Imputación de nulos...", flush=True)
    df_reg["Attendance (%)"] = df_reg["Attendance (%)"].fillna(df_reg["Attendance (%)"].median())
    df_reg["Parent_Education_Level"] = df_reg["Parent_Education_Level"].fillna("Unknown")

    print("3. Filtro IQR de outliers...", flush=True)
    cols_iqr = ["Study_Hours_per_Week", "Attendance (%)", "Sleep_Hours_per_Night", "Age"]

    def limites_iqr(serie):
        q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    mask_validos = pd.Series(True, index=df_reg.index)
    for col in cols_iqr:
        lim_inf, lim_sup = limites_iqr(df_reg[col])
        mask_validos &= df_reg[col].between(lim_inf, lim_sup)

    df_reg = df_reg[mask_validos].reset_index(drop=True)
    print(f"   Registros tras filtro IQR: {len(df_reg)}", flush=True)

    print("4. Codificación de categóricas...", flush=True)
    binary_map = {"Yes": 1, "No": 0}
    df_reg["Extracurricular_Activities"] = df_reg["Extracurricular_Activities"].map(binary_map)
    df_reg["Internet_Access_at_Home"] = df_reg["Internet_Access_at_Home"].map(binary_map)

    categorical_cols = ["Gender", "Department", "Parent_Education_Level", "Family_Income_Level"]
    df_encoded = pd.get_dummies(df_reg, columns=categorical_cols, drop_first=True)

    X_full = df_encoded.drop(columns=[TARGET])
    y_full = df_encoded[TARGET]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=RANDOM_STATE
    )

    print("5. Selección de características según correlación...", flush=True)
    train_con_target = X_train_raw.copy()
    train_con_target[TARGET] = y_train.values

    corr_matrix = train_con_target.corr(numeric_only=True)
    corr_con_target = corr_matrix[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)

    UMBRAL_CORR = 0.05
    seleccionadas = corr_con_target[corr_con_target.abs() >= UMBRAL_CORR].index.tolist()

    if len(seleccionadas) == 0:
        seleccionadas = corr_con_target.abs().sort_values(ascending=False).head(6).index.tolist()

    FEATURES_FINALES = seleccionadas
    print(f"   Features seleccionadas: {FEATURES_FINALES}", flush=True)

    X_train = X_train_raw[FEATURES_FINALES].copy()
    X_test = X_test_raw[FEATURES_FINALES].copy()

    print("6. Escalado StandardScaler...", flush=True)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("7. Entrenamiento de modelos...", flush=True)
    # Modelo 1: Regresión Lineal
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train_scaled, y_train)

    # Modelo 2: Polinomial + Ridge / Lasso
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}

    grid_ridge = GridSearchCV(Ridge(random_state=RANDOM_STATE), param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=1)
    grid_ridge.fit(X_train_poly, y_train)

    grid_lasso = GridSearchCV(Lasso(random_state=RANDOM_STATE, max_iter=10000), param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=1)
    grid_lasso.fit(X_train_poly, y_train)

    if -grid_ridge.best_score_ <= -grid_lasso.best_score_:
        modelo_poly = grid_ridge.best_estimator_
        nombre_reg = f"Ridge (alpha={grid_ridge.best_params_['alpha']})"
    else:
        modelo_poly = grid_lasso.best_estimator_
        nombre_reg = f"Lasso (alpha={grid_lasso.best_params_['alpha']})"

    def eval_model(m, X_tr, X_te):
        pred_te = m.predict(X_te)
        return np.sqrt(mean_squared_error(y_test, pred_te)), mean_absolute_error(y_test, pred_te)

    rmse_lin, mae_lin = eval_model(modelo_lineal, X_train_scaled, X_test_scaled)
    rmse_poly, mae_poly = eval_model(modelo_poly, X_train_poly, X_test_poly)

    if rmse_lin <= rmse_poly:
        mejor_modelo = modelo_lineal
        usa_poly = False
        nombre_mejor = "LinearRegression"
        mae_test_mejor = mae_lin
    else:
        mejor_modelo = modelo_poly
        usa_poly = True
        nombre_mejor = f"PolynomialFeatures(2) + {nombre_reg}"
        mae_test_mejor = mae_poly

    # Salvaguarda: si el modelo regularizado colapsa todos los coeficientes a 0 (modelo degenerado),
    # usamos la Regresión Lineal simple como respaldo para que los sliders/aguja tengan respuesta interactiva.
    es_degenerado = False
    if hasattr(mejor_modelo, "coef_"):
        if np.all(np.abs(mejor_modelo.coef_) < 1e-5):
            es_degenerado = True

    if es_degenerado:
        print(f"\n   [ALERTA] El modelo {nombre_mejor} tiene todos sus coeficientes en 0 (modelo degenerado).", flush=True)
        print("   Activando modelo de respaldo: LinearRegression (para preservar la respuesta dinámica de la aguja).", flush=True)
        mejor_modelo = modelo_lineal
        usa_poly = False
        nombre_mejor = "LinearRegression (Respaldo)"
        mae_test_mejor = mae_lin

    print(f"\n   Mejor modelo seleccionado: {nombre_mejor}", flush=True)
    print(f"   MAE Test: {mae_test_mejor:.4f}", flush=True)

    print("8. Guardando artefactos para Total Score...", flush=True)
    joblib.dump(mejor_modelo, "total_score_model.pkl")
    joblib.dump(scaler, "total_score_scaler.pkl")
    joblib.dump(poly if usa_poly else None, "total_score_poly.pkl")
    joblib.dump(FEATURES_FINALES, "total_score_features.pkl")
    joblib.dump(float(mae_test_mejor), "total_score_mae.pkl")

    print("Artefactos de Total Score guardados exitosamente.", flush=True)
