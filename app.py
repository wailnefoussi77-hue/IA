import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def train_model():
    """
    Train the deep learning model on simulated water-quality data.

    Returns
    -------
    tf.keras.Model
        The trained TensorFlow model ready for predictions.
    float
        The accuracy of the model evaluated on a held-out test set.
    """
    # Generate synthetic dataset to simulate water quality tests
    np.random.seed(42)
    n_samples = 10_000
    fer = np.random.uniform(0.0, 1.0, n_samples)
    nitrates = np.random.uniform(0.0, 100.0, n_samples)
    chlorures = np.random.uniform(0.0, 500.0, n_samples)
    potable = ((fer < 0.3) & (nitrates < 50) & (chlorures < 250)).astype(int)

    df = pd.DataFrame({
        "Fer_mgL": fer,
        "Nitrates_mgL": nitrates,
        "Chlorures_mgL": chlorures,
        "Potable": potable,
    })
    X = df[["Fer_mgL", "Nitrates_mgL", "Chlorures_mgL"]]
    y = df["Potable"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build the neural network
    model = Sequential([
        Dense(16, activation="relu", input_shape=(3,)),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # Train the model
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    # Evaluate accuracy on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return model, test_acc


def main():
    st.set_page_config(
        page_title="Prédiction de la potabilité de l'eau",
        layout="centered",
    )
    st.title("Prédiction de la potabilité de l'eau par IA")

    st.markdown(
        """
        Cette application utilise un modèle de Deep Learning pour déterminer si une eau est potable ou non
        en fonction de la concentration en fer, nitrates et chlorures. Les données d'entraînement sont
        simulées à partir des normes de l'Organisation mondiale de la santé (OMS).
        """
    )

    # Train (or load) the model
    with st.spinner("Entraînement du modèle en cours..."):
        model, accuracy = train_model()
    st.success(f"Modèle entraîné avec une précision de {accuracy * 100:.2f} % sur des données de test")

    st.header("Saisissez une nouvelle analyse d'eau")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_fer = st.number_input(
            "Fer (mg/L)", min_value=0.0, step=0.01, value=0.1
        )
    with col2:
        test_nitrates = st.number_input(
            "Nitrates (mg/L)", min_value=0.0, step=0.1, value=20.0
        )
    with col3:
        test_chlorures = st.number_input(
            "Chlorures (mg/L)", min_value=0.0, step=0.1, value=100.0
        )

    if st.button("Analyser"):
        # Prepare input for prediction
        new_sample = pd.DataFrame({
            "Fer_mgL": [test_fer],
            "Nitrates_mgL": [test_nitrates],
            "Chlorures_mgL": [test_chlorures],
        })
        proba = model.predict(new_sample, verbose=0)[0][0]
        result = int(proba > 0.5)
        st.write(f"Probabilité que l'eau soit potable : **{proba * 100:.2f} %**")
        if result == 1:
            st.success("L'IA estime que l'eau est **potable** ✅")
        else:
            st.error("L'IA estime que l'eau est **non potable** ❌")
            st.markdown("**Raison(s) possible(s) :**")
            if test_fer >= 0.3:
                st.markdown(f"- Fer trop élevé : {test_fer:.2f} mg/L ≥ 0.3 mg/L")
            if test_nitrates >= 50:
                st.markdown(
                    f"- Nitrates trop élevés : {test_nitrates:.2f} mg/L ≥ 50 mg/L"
                )
            if test_chlorures >= 250:
                st.markdown(
                    f"- Chlorures trop élevés : {test_chlorures:.2f} mg/L ≥ 250 mg/L"
                )


if __name__ == "__main__":
    main()