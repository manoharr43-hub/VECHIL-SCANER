import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch
from datetime import datetime

# =============================
# PAGE SETUP
# =============================
st.set_page_config(page_title="AI Vehicle Scanner PRO MAX", layout="wide")
st.title("🚗 AI Vehicle Scanner + Damage Detection")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        return YOLO("yolov8n.pt")

model = load_model()
CLASS_NAMES = model.names

# =============================
# SETTINGS
# =============================
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 5)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# DAMAGE KEYWORDS (FIXED ✅)
# =============================
damage_keywords = ["scratch", "dent", "crack", "burn", "broken"]

# =============================
# OPTIONAL SENSOR (SAFE MODE)
# =============================
def get_sensor_status():
    # Cloud లో sensor ఉండదు → safe
