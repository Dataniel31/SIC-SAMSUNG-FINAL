# Importaciones
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image

# Título de la aplicación
st.title("Clasificación de Neumonía en Radiografías")

# Descripción del proyecto
st.markdown("""
## Descripción del Proyecto
Esta aplicación clasifica radiografías de tórax en dos categorías: NORMAL y PNEUMONIA usando un modelo de CNN.
""")

# Cargar el modelo entrenado
model_path = "modelo_neumonia.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.write("Modelo cargado correctamente.")
else:
    st.write("No se encontró un modelo entrenado. Por favor entrena el modelo primero.")

# Paso 1: Cargar y mostrar una imagen de ejemplo
st.header("1. Cargar una Imagen de Radiografía")
uploaded_file = st.file_uploader("Elige una imagen de radiografía", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen Cargada", use_column_width=True)
    st.write("Imagen cargada correctamente.")

    # Paso 2: Predecir con la imagen cargada
    st.header("2. Clasificar la Imagen")
    if st.button("Clasificar Imagen"):
        # Preprocesamiento de la imagen cargada
        try:
            # Asegúrate de que la imagen tenga 3 canales
            image_resized = image.convert('RGB').resize((150, 150))  # Redimensiona y convierte a RGB
            img_array = img_to_array(image_resized) / 255.0  # Normaliza la imagen
            img_array = np.expand_dims(img_array, axis=0)  # Agrega la dimensión para el batch

            # Realizar la predicción
            prediction = model.predict(img_array)
            pred_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
            st.write(f"**Predicción**: {pred_class}")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {str(e)}")
