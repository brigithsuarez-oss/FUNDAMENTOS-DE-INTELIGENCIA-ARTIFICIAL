import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo_brigith_fondo.keras"
CLASSES_PATH = BASE_DIR / "clases.json"
IMAGE_SIZE = 224


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_classes():
    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de clases: {CLASSES_PATH}")
    with CLASSES_PATH.open("r", encoding="utf-8") as file:
        classes = json.load(file)
    if not isinstance(classes, list) or not classes:
        raise ValueError("clases.json debe contener una lista de nombres de clase.")
    return classes


st.set_page_config(page_title="Clasificador BRIGITH / Fondo", page_icon="🖼️")
st.title("Clasificador de imágenes")
st.write("Carga una fotografía para saber si el modelo la clasifica como BRIGITH o Fondo.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("No se pudo leer el archivo. Prueba con otra imagen.")
    else:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Imagen seleccionada", use_container_width=True)

        try:
            model = load_model()
            classes = load_classes()
            resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
            batch = np.expand_dims(resized.astype(np.float32), axis=0)
            prediction = model.predict(preprocess_input(batch), verbose=0)[0]

            if len(prediction) != len(classes):
                raise ValueError(
                    f"El modelo devuelve {len(prediction)} clases, "
                    f"pero clases.json contiene {len(classes)}."
                )

            best_index = int(np.argmax(prediction))
            st.subheader(f"Predicción: {classes[best_index]}")
            st.metric("Confianza", f"{prediction[best_index]:.1%}")

            st.write("Probabilidad por clase:")
            for class_name, probability in zip(classes, prediction):
                st.write(f"{class_name}: {probability:.1%}")
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))
