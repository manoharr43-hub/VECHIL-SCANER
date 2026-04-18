import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch

# ============================= #
# CONFIG
# ============================= #
st.set_page_config(page_title="Vehicle AI Scanner PRO+", layout="wide")
st.title("🚗 Vehicle AI Scanner (ULTRA STABLE + FAST)")

# ============================= #
# STATE
# ============================= #
if "run" not in st.session_state:
    st.session_state.run = False

# ============================= #
# DEVICE SETUP (CPU/GPU)
# ============================= #
device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Using device: **{device}**")

# ============================= #
# LOAD MODEL
# ============================= #
@st.cache_resource
def load_model(model_path: str = "best.pt"):
    model = YOLO(model_path)
    model.to(device)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()

CLASS_NAMES = model.names

# ============================= #
# UPLOAD
# ============================= #
video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start") and video is not None:
        st.session_state.run = True
with col2:
    if st.button("🛑 Stop"):
        st.session_state.run = False

if video is None:
    st.info("Please upload a video to start.")
    st.stop()

# ============================= #
# PROCESSING
# ============================= #
if st.session_state.run:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        os.remove(temp_path)
        st.stop()

    stframe = st.empty()
    progress = st.progress(0)
    logs = []
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    st.info("🚀 Processing started...")
    skip_frames = 3  # process every 3rd frame for speed

    try:
