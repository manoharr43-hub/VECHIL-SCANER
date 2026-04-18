import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Vehicle AI Scanner PRO+", layout="wide")
st.title("🚗 Vehicle AI Scanner (ULTRA STABLE + FAST)")

# =============================
# STATE
# =============================
if "run" not in st.session_state:
    st.session_state.run = False

# =============================
# DEVICE SETUP (GPU BOOST)
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# LOAD MODEL (OPTIMIZED)
# =============================
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    model.to(device)
    return model

model = load_model()
CLASS_NAMES = model.names

# =============================
# UPLOAD
# =============================
video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start"):
        st.session_state.run = True

with col2:
    if st.button("🛑 Stop"):
        st.session_state.run = False

# =============================
# PROCESSING
# =============================
if video and st.session_state.run:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video.read())
    temp_path = temp.name

    cap = cv2.VideoCapture(temp_path)

    stframe = st.empty()
    progress = st.progress(0)

    logs = []
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    st.info("🚀 Processing Started...")

    # SPEED OPTIMIZATION
    skip_frames = 3  # increase speed

    while cap.isOpened() and st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        # skip frames for speed
        if frame_id % skip_frames != 0:
            frame_id += 1
            continue

        # =============================
        # YOLO INFERENCE (FAST MODE)
        # =============================
        results = model.predict(
            frame,
            imgsz=512,
            conf=0.4,
            device=device,
            verbose=False
        )

        r = results[0]
        output = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.data.cpu().numpy()

            for box in boxes:
                logs.append({
                    "frame": frame_id,
                    "class_name": CLASS_NAMES[int(box[5])],
                    "confidence": float(box[4]),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                })

        # UI update
        if frame_id % 10 == 0:
            stframe.image(output, channels="BGR", use_container_width=True)

        frame_id += 1

        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

        # safety stop
        if frame_id > 1000:
            st.warning("⚠️ Frame limit reached (safe mode)")
            break

    cap.release()
    os.remove(temp_path)

    st.success("✅ Processing Completed")

    # =============================
    # REPORT
    # =============================
    df = pd.DataFrame(logs)

    st.subheader("📊 Detection Report")

    if not df.empty:
        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False).encode(),
            "vehicle_report.csv",
            "text/csv"
        )
    else:
        st.warning("No objects detected.")
