import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")
st.title("❤️ Heart Disease Prediction Dashboard")

# 1. Load the Results and Scaler
try:
    results_df = pd.read_csv("model_results.csv")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error("Missing files. Run 'python model_training.py' first.")
    st.stop()

# 2. Sidebar Dropdown
st.sidebar.header("Model Selection")
selected_model_name = st.sidebar.selectbox("Choose a Model", results_df["ML Model Name"])

# 3. Display Metrics
st.subheader(f"📊 {selected_model_name} Metrics")
m = results_df[results_df["ML Model Name"] == selected_model_name].iloc[0]
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{m['Accuracy']:.2%}")
col2.metric("AUC", f"{m['AUC']:.2f}")
col3.metric("Precision", f"{m['Precision']:.2f}")
col4.metric("Recall", f"{m['Recall']:.2f}")
col5.metric("F1", f"{m['F1']:.2f}")
col6.metric("MCC", f"{m['MCC']:.2f}")

# 4. Dataset Upload
st.divider()
st.subheader("📂 Prediction & Evaluation")
uploaded_file = st.file_uploader("Upload heart.csv", type=["csv"])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    original_df = test_df.copy()

    try:
        # Preprocessing: Load Encoders (using lowercase filenames)
        cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for col in cat_cols:
            encoder_path = f"models/{col.lower()}_encoder.pkl"
            le = joblib.load(encoder_path)
            test_df[col] = le.transform(test_df[col].astype(str))

        # Drop Target if present
        X_test = test_df.drop("HeartDisease", axis=1, errors='ignore')
        
        # Scale and Load Model
        X_test_scaled = scaler.transform(X_test)
        model_filename = selected_model_name.lower().replace(" ", "_") + ".pkl"
        model = joblib.load(f"models/{model_filename}")

        # Predict
        preds = model.predict(X_test_scaled)
        original_df["Prediction"] = preds
        
        st.write("### Prediction Results", original_df.head())

        # 5. Confusion Matrix
        if "HeartDisease" in original_df.columns:
            st.subheader("📈 Confusion Matrix")
            cm = confusion_matrix(original_df["HeartDisease"], preds)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")