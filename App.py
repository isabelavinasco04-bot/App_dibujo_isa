import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Modelo pre-entrenado para reconocer objetos
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

st.title("Tablero para dibujo con reconocimiento IA")

with st.sidebar:
    st.subheader("Propiedades del Tablero")

    # Canvas dimensions
    st.subheader("Dimensiones del Tablero")
    canvas_width = st.slider("Ancho del tablero", 300, 700, 500, 50)
    canvas_height = st.slider("Alto del tablero", 200, 600, 300, 50)

    # Drawing mode selector
    drawing_mode = st.selectbox(
        "Herramienta de Dibujo:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )

    # Stroke width slider
    stroke_width = st.slider("Selecciona el ancho de línea", 1, 30, 15)

    # Stroke color picker
    stroke_color = st.color_picker("Color de trazo", "#FFFFFF")

    # Background color
    bg_color = st.color_picker("Color de fondo", "#000000")

# Canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=canvas_height,
    width=canvas_width,
    drawing_mode=drawing_mode,
    key=f"canvas_{canvas_width}_{canvas_height}",
)

# --- Procesar lo dibujado y hacer predicción ---
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)

    # Preprocesar para el modelo
    img_resized = np.array(st.image(img, caption="Tu dibujo", use_container_width=True))
    img_resized = np.array(img)
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = preprocess_input(img_resized)

    # Predicción
    preds = model.predict(img_resized)
    decoded = decode_predictions(preds, top=1)[0]

    st.subheader("Lo que dibujaste podría ser:")
    st.write(f"👉 {decoded[0][1]} (con {decoded[0][2]*100:.2f}% de confianza)")
