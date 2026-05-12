import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

st.title("Prédiction de la potabilité de l'eau par IA")

st.write(
    "Cette IA prédit si une eau est potable ou non selon plusieurs paramètres chimiques."
)

# Fonction mise en cache : elle évite de réentraîner le modèle à chaque clic
@st.cache_resource
def entrainer_modele():
    np.random.seed(42)
    n_samples = 10000

    fer = np.random.uniform(0.0, 1.0, n_samples)
    nitrates = np.random.uniform(0.0, 100.0, n_samples)
    chlorures = np.random.uniform(0.0, 500.0, n_samples)

    potable = (
        (fer < 0.3)
        & (nitrates < 50)
        & (chlorures < 250)
    ).astype(int)

    df = pd.DataFrame({
        "Fer_mgL": fer,
        "Nitrates_mgL": nitrates,
        "Chlorures_mgL": chlorures,
        "Potable": potable
    })

    X = df[["Fer_mgL", "Nitrates_mgL", "Chlorures_mgL"]]
    y = df["Potable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    precision = accuracy_score(y_test, predictions)

    return model, precision


with st.spinner("Chargement du modèle IA..."):
    model, precision = entrainer_modele()

st.success(f"Précision du modèle : {precision * 100:.2f}%")

st.header("Tester une nouvelle eau")

test_fer = st.number_input(
    "Fer (mg/L)",
    min_value=0.0,
    max_value=10.0,
    value=0.10
)

test_nitrates = st.number_input(
    "Nitrates (mg/L)",
    min_value=0.0,
    max_value=500.0,
    value=20.0
)

test_chlorures = st.number_input(
    "Chlorures (mg/L)",
    min_value=0.0,
    max_value=1000.0,
    value=100.0
)

if st.button("Analyser"):

    nouvelle_eau = pd.DataFrame({
        "Fer_mgL": [test_fer],
        "Nitrates_mgL": [test_nitrates],
        "Chlorures_mgL": [test_chlorures]
    })

    prediction = model.predict(nouvelle_eau)[0]
    proba = model.predict_proba(nouvelle_eau)[0][1]

    st.write(f"Probabilité de potabilité : {proba * 100:.2f}%")

    if prediction == 1:
        st.success("Eau POTABLE ✅")

    else:
        st.error("Eau NON POTABLE ❌")

        if test_fer >= 0.3:
            st.write("⚠️ Fer supérieur au seuil OMS")

        if test_nitrates >= 50:
            st.write("⚠️ Nitrates supérieurs au seuil OMS")

        if test_chlorures >= 250:
            st.write("⚠️ Chlorures supérieurs au seuil OMS")