# Importaciones
import streamlit as st
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

# Título de la aplicación
st.title("Clasificación de Neumonía en Radiografías")

# Descripción del proyecto
st.markdown("""
## Descripción del Proyecto
Esta aplicación clasifica radiografías de tórax en dos categorías: NORMAL y PNEUMONIA usando un modelo de CNN.
""")

# Paso 1: Cargar y mostrar una imagen de ejemplo
st.header("1. Cargar una Imagen de Radiografía")
uploaded_file = st.file_uploader("Elige una imagen de radiografía", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen Cargada", use_column_width=True)
    st.write("Imagen cargada correctamente.")

# Paso 2: Definir y cargar el modelo (si no está entrenado)
st.header("2. Entrenar o Cargar el Modelo")
if st.button("Entrenar el Modelo"):
    # Definición del modelo de CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    st.write("Modelo definido y compilado.")

    # Entrenamiento simulado (en un entorno de producción, agregaría aquí `model.fit`)
    st.write("El modelo ha sido entrenado (aquí deberías entrenarlo realmente).")
    
    # Guardar el modelo
    model.save("modelo_neumonia.h5")
else:
    # Cargar el modelo entrenado
    if os.path.exists("modelo_neumonia.h5"):
        model = load_model("modelo_neumonia.h5")
        st.write("Modelo cargado.")
    else:
        st.write("No se encontró un modelo entrenado. Entrena el modelo primero.")

# Paso 3: Predecir con la imagen cargada
st.header("3. Clasificación de la Imagen")
if st.button("Clasificar Imagen") and uploaded_file is not None:
    # Preprocesamiento de la imagen cargada
    img_array = img_to_array(image.resize((150, 150))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    prediction = model.predict(img_array)
    pred_class = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    st.write(f"**Predicción**: {pred_class}")

    # Mostrar la probabilidad de neumonía
    st.write(f"Probabilidad de neumonía: {prediction[0][0]:.2f}")
