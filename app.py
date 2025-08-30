import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('brain_tumor_custom_cnn.h5')

# Define the class names as per your training
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit app configuration
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain Tumor MRI Image Classifier")
st.markdown("Upload an MRI image to classify the tumor type (Glioma, Meningioma, Pituitary, or No Tumor).")

# File uploader for MRI image
uploaded_file = st.file_uploader("Choose an MRI Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess for model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict tumor class
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🔍 Predicted Tumor Type: **{predicted_class.upper()}**")
    st.bar_chart(prediction[0])

# Separator
st.markdown("---")

# Batch Prediction for multiple images
st.markdown("#### 📂 Batch Prediction (Optional)")
st.info("This feature supports batch prediction of multiple MRI images (coming soon).")
