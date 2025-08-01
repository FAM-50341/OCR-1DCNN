import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from utils.ocr_utils import extract_text_from_image, parse_cbc_features

# Load model
model = tf.keras.models.load_model("model/thalassemia_1dcnn_model.h5")

class_labels = ["Normal", "Silent Carrier", "Trait", "Disease"]

st.set_page_config(page_title="CBC → Thalassemia Classifier", layout="centered")
st.title("🧬 CBC Report → Thalassemia Prediction")

uploaded_file = st.file_uploader("📄 Upload CBC Report Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded CBC Report", use_column_width=True)

    with st.spinner("🔍 Performing OCR..."):
        text = extract_text_from_image(image)
        st.text_area("📃 Extracted Text", text)

    with st.spinner("🔬 Parsing CBC Features..."):
        features = parse_cbc_features(text)
        st.write("🔢 Parsed Features:", features)

    # Preprocess for 1D CNN: (1, 9, 1)
    input_array = np.array(features).reshape(1, 9, 1)

    with st.spinner("🤖 Predicting..."):
        prediction = model.predict(input_array)
        predicted_class = np.argmax(prediction)

        st.success(f"🩺 Predicted Class: **{class_labels[predicted_class]}**")
        st.write("🔎 Probabilities:", {label: float(p) for label, p in zip(class_labels, prediction[0])})
