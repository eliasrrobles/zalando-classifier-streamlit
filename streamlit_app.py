import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
 
model = keras.models.load_model("models/zalando.keras")
 
class_names = [
    "Camiseta/top",
    "Pantalón",
    "Sudadera",
    "Vestido",
    "Abrigo",
    "Sandalia",
    "Camisa",
    "Zapatillas",
    "Bolso",
    "Botas"
]

st.title("Ialando - Clasificación de prendas")
 
uploaded_file = st.file_uploader(
    "Sube una imagen de la prenda que quieres clasificar",
    type=["jpg", "png", "jpeg"]
)
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L") # Escala de grises
    image = ImageOps.invert(image) # Invierte la imagen
    image = image.resize((28, 28)) # Redimensiona
    image = ImageOps.autocontrast(image)
    st.image(image, caption="Imagen procesada")

    # Convierte la imagen a un array normalizado
    img_array = (np.array(image, dtype=np.float32) / 255.0).flatten()
    #img_array = img_array.reshape(1, 28, 28, 1)

    #Predicción
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Muestra el resultado
    st.write(f"Vector de predicción: {prediction}")
    st.subheader(f"Predicción hecha por la IA: {predicted_class}")
