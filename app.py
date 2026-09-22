import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection using Deep Learning")
st.write("Upload a plant leaf image to detect its disease.")

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/plant_disease_model.keras")

model = load_model()

# -----------------------------------
# Load Class Names
# -----------------------------------
CLASS_NAMES = sorted(os.listdir("dataset/PlantVillage/train"))

# -----------------------------------
# Upload Image
# -----------------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------------
    # Preprocess Image
    # -----------------------------------
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    # Same preprocessing used during training
    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # -----------------------------------
    # Prediction
    # -----------------------------------
    prediction = model.predict(img, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    plant, disease = predicted_class.split("___")

    # -----------------------------------
    # Results
    # -----------------------------------
    st.success(f"🌱 Plant : **{plant}**")
    st.success(f"🦠 Disease : **{disease}**")
    st.info(f"🎯 Confidence : **{confidence:.2f}%**")

    # -----------------------------------
    # Confidence Bar
    # -----------------------------------
    st.progress(float(confidence / 100))

    # -----------------------------------
    # Top 5 Predictions
    # -----------------------------------
    st.subheader("Top 5 Predictions")

    top5 = np.argsort(prediction)[-5:][::-1]

    for i in top5:
        st.write(f"**{CLASS_NAMES[i]}** : {prediction[i] * 100:.2f}%")