%%writefile app.py
import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

st.title("🩺 Chest X-ray Pneumonia Detection (ML Baseline)")
st.write("Upload an X-ray image and get the model prediction with confidence level.")

# Automatically find the uploaded .pkl model
model_file = next(iter([f for f in os.listdir('.') if f.endswith('.pkl')]))
model = joblib.load(model_file)

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess for ML model (must match training setup)
    image = image.resize((64, 64))
    img_array = np.array(image).flatten() / 255.0
    img_array = img_array.reshape(1, -1)  # (1, 4096)

    # Predict
    pred = model.predict(img_array)[0]
    label = "Normal" if pred == 0 else "Pneumonia"

    # Confidence (if model supports predict_proba)
    if hasattr(model, "predict_proba"):
        pred_prob = model.predict_proba(img_array)[0][1] if pred == 1 else model.predict_proba(img_array)[0][0]
        st.write(f"Predicted Class: **{label}**")
        st.write(f"Confidence: {pred_prob*100:.2f}%")
    else:
        st.write(f"Predicted Class: **{label}**")
        st.warning("⚠️ Confidence not available (model has no predict_proba).")
