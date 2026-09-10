"""
Script para entrenar el modelo de Clasificación Múltiple y generar los artefactos .pkl
necesarios para la interfaz Streamlit (app.py).
"""

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

RANDOM_STATE = 42

if __name__ == "__main__":
    print("1. Cargando dataset 'Students_Grading_Dataset_Biased.csv'...", flush=True)
    df = pd.read_csv("Students_Grading_Dataset_Biased.csv")
    order = ["A", "B", "C", "D", "F"]

    print(f"   Dimensiones originales: {df.shape}", flush=True)

    print("2. Limpieza e imputación de nulos...", flush=True)
    df_clean = df.copy()

    for col in ["Attendance (%)", "Assignments_Avg"]:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"   {col}: imputado con mediana = {median_val:.2f}", flush=True)

    df_clean["Parent_Education_Level"] = df_clean["Parent_Education_Level"].fillna("Unknown")

    print("3. Codificación de variables...", flush=True)
    binary_map = {"Yes": 1, "No": 0}
    df_clean["Extracurricular_Activities"] = df_clean["Extracurricular_Activities"].map(binary_map)
    df_clean["Internet_Access_at_Home"] = df_clean["Internet_Access_at_Home"].map(binary_map)

    categorical_cols = ["Gender", "Department", "Parent_Education_Level", "Family_Income_Level"]
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    drop_cols = ["Student_ID", "First_Name", "Last_Name", "Email"]
    df_encoded = df_encoded.drop(columns=drop_cols)

    print(f"   Dataset procesado: {df_encoded.shape}", flush=True)

    X = df_encoded.drop(columns=["Grade"])
    y = df_encoded["Grade"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   Train: {X_train.shape} | Test: {X_test.shape}", flush=True)

    print("4. Escalado de variables (StandardScaler)...", flush=True)
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("5. Optimización de hiperparámetros (GridSearchCV para C)...", flush=True)
    param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
    base_model = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    grid = GridSearchCV(base_model, param_grid, cv=5, scoring="f1_macro", n_jobs=1)
    grid.fit(X_train_scaled, y_train)

    best_C = grid.best_params_["C"]
    print(f"   Mejor C encontrado: {best_C}", flush=True)

    print("6. Entrenamiento del modelo final...", flush=True)
    model = LogisticRegression(
        solver="lbfgs",
        C=best_C,
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )
    model.fit(X_train_scaled, y_train)

    # Evaluación
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    _, _, f1_train, _ = precision_recall_fscore_support(y_train, y_train_pred, average="macro", zero_division=0)
    _, _, f1_test, _ = precision_recall_fscore_support(y_test, y_test_pred, average="macro", zero_division=0)

    print(f"\n--- Métricas del Experimento ---", flush=True)
    print(f"Train - Accuracy: {acc_train:.4f} | F1-macro: {f1_train:.4f}", flush=True)
    print(f"Test  - Accuracy: {acc_test:.4f} | F1-macro: {f1_test:.4f}", flush=True)

    print("\n7. Guardando artefactos (.pkl)...", flush=True)
    joblib.dump(model, "grade_model.pkl")
    joblib.dump(scaler, "grade_scaler.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

    print("Artefactos guardados exitosamente ('grade_model.pkl', 'grade_scaler.pkl', 'feature_columns.pkl').", flush=True)
