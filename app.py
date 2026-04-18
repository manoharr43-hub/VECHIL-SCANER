import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch
from datetime import datetime
import serial
import time

# =============================
# PAGE SETUP
# =============================
st.set_page_config(page_title="AI Vehicle Scanner PRO MAX", layout="wide")
st.title("🚗⚡ AI Vehicle Scanner + Electrical Diagnostic System")

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
# SIDEBAR SETTINGS
# =============================
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.3)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, 5)

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# SERIAL CONNECTION (ESP32)
# =============================
st.sidebar.subheader("🔌 Electrical Sensor")

port = st.sidebar.text_input("COM Port", "COM3")
connect_btn = st.sidebar.button("Connect Sensor")

ser = None

if connect_btn:
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        st.sidebar.success("✅ Sensor Connected")
        time.sleep(2)
    except:
        st.sidebar.error("❌ Connection Failed")

def get_sensor_status():
    if ser:
        try:
            data = ser.readline().decode().strip()
            return data
        except:
            return "NO DATA"
    return "DISCONNECTED"

# =============================
# DAMAGE KEYWORDS
# =============================
damage_keywords = ["scratch", "dent", "crack", "burn
