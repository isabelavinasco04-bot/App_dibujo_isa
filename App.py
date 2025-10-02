import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

# Intentamos cargar OpenCV
try:
    import cv2
except ModuleNotFoundError:
    st.error("Falta OpenCV. Agrega 'opencv-python-headless' a requirements.txt y vuelve a desplegar.")
    st.stop()

st.title("Tablero para dibujo + reconocimiento de figuras (casa, corazón, estrella)")

# ------------------ UI ------------------
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

# ------------------ Plantillas "semánticas" ------------------

def template_heart(size=256):
    """Genera máscara binaria de un corazón centrado."""
    img = np.zeros((size, size), dtype=np.uint8)
    # Ecuación paramétrica del corazón
    t = np.linspace(0, 2*np.pi, 800)
    x = 16 * (np.sin(t) ** 3)
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    pts = np.stack([x*(size*0.7)+size*0.15, (1-y)*(size*0.7)+size*0.15], axis=1).astype(np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

def template_house(size=256):
    """Genera máscara binaria de una casa (cuerpo + techo)."""
    img = np.zeros((size, size), dtype=np.uint8)
    # Cuerpo (rectángulo)
    body = np.array([
        [int(size*0.25), int(size*0.55)],
        [int(size*0.75), int(size*0.55)],
        [int(size*0.75), int(size*0.85)],
        [int(size*0.25), int(size*0.85)],
    ], dtype=np.int32)
    # Techo (triángulo)
    roof = np.array([
        [int(size*0.20), int(size*0.55)],
        [int(size*0.80), int(size*0.55)],
        [int(size*0.50), int(size*0.20)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [body], 255)
    cv2.fillPoly(img, [roof], 255)
    return img

def template_star(size=256, points=5, inner_ratio=0.45):
    """Genera máscara binaria de una estrella regular."""
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size//2, size//2
    R = int(size*0.38)
    r = int(R*inner_ratio)
    angles = np.linspace(0, 2*np.pi, points*2, endpoint=False) - np.pi/2
    pts = []
    for i, ang in enumerate(angles):
        rad = R if i % 2 == 0 else r
        x = int(cx + rad*np.cos(ang))
        y = int(cy + rad*np.sin(ang))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], 255)
    return img

@st.cache_resource
def build_templates():
    """Devuelve contornos normalizados de las plantillas."""
    templates = {
        "corazón": template_heart(),
        "casa": template_house(),
        "estrella": template_star(),
    }
    contours = {}
    for name, mask in templates.items():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

