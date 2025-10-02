import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

# ---- Dependencias (OpenCV + scikit-learn) ----
try:
    import cv2
except ModuleNotFoundError:
    st.error("Falta OpenCV. Agrega 'opencv-python-headless' a requirements.txt y vuelve a desplegar.")
    st.stop()

try:
    from sklearn.neighbors import KNeighborsClassifier
except ModuleNotFoundError:
    st.error("Falta scikit-learn. Agrega 'scikit-learn' a requirements.txt y vuelve a desplegar.")
    st.stop()

st.set_page_config(page_title="Tablero IA", layout="centered")
st.title("Tablero para dibujo + IA (con botón Predecir)")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader("Propiedades del Tablero")

    st.subheader("Dimensiones del Tablero")
    canvas_width = st.slider("Ancho del tablero", 300, 700, 500, 50)
    canvas_height = st.slider("Alto del tablero", 200, 600, 300, 50)

    drawing_mode = st.selectbox(
        "Herramienta de Dibujo:",
        ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
    )

    stroke_width = st.slider("Selecciona el ancho de línea", 1, 30, 15)
    stroke_color = st.color_picker("Color de trazo", "#FFFFFF")
    bg_color = st.color_picker("Color de fondo", "#000000")

# -------------------- Canvas --------------------
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

# -------------------- Estado (dataset y modelo) --------------------
if "X" not in st.session_state:
    st.session_state.X = []  # vectores de características
if "y" not in st.session_state:
    st.session_state.y = []  # etiquetas
if "model" not in st.session_state:
    st.session_state.model = None
if "n_neighbors" not in st.session_state:
    st.session_state.n_neighbors = 3

# -------------------- Utilidades --------------------
def preprocess_to_vector(img_rgba: np.ndarray, target_size: int = 28) -> np.ndarray:
    """
    Toma la imagen RGBA del canvas, la binariza, recorta el bounding box del trazo,
    la redimensiona a target_size x target_size y devuelve un vector normalizado [0,1].
    """
    img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertir si el fondo quedó blanco (o hay poco contraste)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    # Limpieza ligera
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Si no hay tinta suficiente, devolvemos vector vacío
    ink = np.where(th > 0)
    if len(ink[0]) == 0:
        return None

    # Bounding box del trazo con margen
    y0, y1 = np.min(ink[0]), np.max(ink[0])
    x0, x1 = np.min(ink[1]), np.max(ink[1])
    pad = 4
    y0 = max(y0 - pad, 0)
    x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, th.shape[0] - 1)
    x1 = min(x1 + pad, th.shape[1] - 1)

    crop = th[y0:y1+1, x0:x1+1]

    # Redimensionar a cuadrado target_size
    resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalizar a [0,1] y aplanar
    vec = (resized.astype(np.float32) / 255.0).reshape(-1)
    return vec

def basic_shape_classifier(img_rgba: np.ndarray) -> str:
    """
    Clasificador heurístico de formas básicas: círculo, triángulo, cuadrado/rectángulo, línea, polígono, punto, nada.
    """
    if img_rgba is None:
        return "nada"

    img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(th) > 127:
        th = cv2.bitwise_not(th)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv


