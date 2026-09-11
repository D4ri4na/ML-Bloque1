"""
Script para entrenar el modelo de Regresión sobre el dataset UCI student-mat.csv (G3 de 0 a 20)
y generar los artefactos .pkl necesarios para la interfaz Streamlit (app.py).
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
    print("1. Cargando dataset UCI 'dataset/student-mat.csv' (sep=';')...", flush=True)
    df = pd.read_csv("dataset/student-mat.csv", sep=";")
    print(f"   Dimensiones originales: {df.shape}", flush=True)

    TARGET = "G3"
    # Prevenir Data Leakage: G2 y G3 fuera de X; se conserva G1 (nota 1er parcial, 0-20) y comportamiento
    cols_a_eliminar = ["G2", "G3"]
    X_full_df = df.drop(columns=cols_a_eliminar)
    y_full = df[TARGET]

    print("2. Codificación de variables categóricas (get_dummies)...", flush=True)
    cat_cols = X_full_df.select_dtypes(include=["object", "string"]).columns.tolist()
    X_encoded = pd.get_dummies(X_full_df, columns=cat_cols, drop_first=True)

    print("3. División en conjunto de Entrenamiento (80%) y Prueba (20%)...", flush=True)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_encoded, y_full, test_size=0.2, random_state=RANDOM_STATE
    )

    print("4. Selección de características según correlación (|r| >= 0.10)...", flush=True)
    train_con_target = X_train_raw.copy()
    train_con_target[TARGET] = y_train.values

    corr_matrix = train_con_target.corr(numeric_only=True)
    corr_con_target = corr_matrix[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)

    UMBRAL_CORR = 0.10
    seleccionadas = corr_con_target[corr_con_target.abs() >= UMBRAL_CORR].index.tolist()

    if len(seleccionadas) == 0:
        seleccionadas = corr_con_target.abs().sort_values(ascending=False).head(6).index.tolist()

    FEATURES_FINALES = seleccionadas
    print(f"   Features seleccionadas ({len(FEATURES_FINALES)}): {FEATURES_FINALES}", flush=True)

    X_train = X_train_raw[FEATURES_FINALES].copy()
    X_test = X_test_raw[FEATURES_FINALES].copy()

    print("5. Escalado StandardScaler...", flush=True)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("6. Entrenamiento de modelos (Lineal vs Polinomial + Ridge/Lasso)...", flush=True)
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

    def eval_model(m, X_te):
        pred_te = m.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_test, pred_te))
        mae = mean_absolute_error(y_test, pred_te)
        r2 = r2_score(y_test, pred_te)
        return rmse, mae, r2

    rmse_lin, mae_lin, r2_lin = eval_model(modelo_lineal, X_test_scaled)
    rmse_poly, mae_poly, r2_poly = eval_model(modelo_poly, X_test_poly)

    if rmse_lin <= rmse_poly:
        mejor_modelo = modelo_lineal
        usa_poly = False
        nombre_mejor = "LinearRegression"
        mae_test_mejor = mae_lin
        r2_test_mejor = r2_lin
    else:
        mejor_modelo = modelo_poly
        usa_poly = True
        nombre_mejor = f"PolynomialFeatures(2) + {nombre_reg}"
        mae_test_mejor = mae_poly
        r2_test_mejor = r2_poly

    # Salvaguarda: si el modelo regularizado anula todos los coeficientes
    es_degenerado = False
    if hasattr(mejor_modelo, "coef_"):
        if np.all(np.abs(mejor_modelo.coef_) < 1e-5):
            es_degenerado = True

    if es_degenerado:
        print(f"\n   [ALERTA] El modelo {nombre_mejor} tiene todos sus coeficientes en 0 (modelo degenerado).", flush=True)
        print("   Activando modelo de respaldo: LinearRegression.", flush=True)
        mejor_modelo = modelo_lineal
        usa_poly = False
        nombre_mejor = "LinearRegression (Respaldo)"
        mae_test_mejor = mae_lin
        r2_test_mejor = r2_lin

    print(f"\n   Mejor modelo seleccionado: {nombre_mejor}", flush=True)
    print(f"   MAE Test: {mae_test_mejor:.4f} (sobre 20)  |  R2 Test: {r2_test_mejor:.4f}", flush=True)

    print("7. Guardando artefactos para Total Score (0-20)...", flush=True)
    joblib.dump(mejor_modelo, "total_score_model.pkl")
    joblib.dump(scaler, "total_score_scaler.pkl")
    joblib.dump(poly if usa_poly else None, "total_score_poly.pkl")
    joblib.dump(FEATURES_FINALES, "total_score_features.pkl")
    joblib.dump(float(mae_test_mejor), "total_score_mae.pkl")

    print("Artefactos guardados exitosamente.", flush=True)
