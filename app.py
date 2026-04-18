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
st.title("🚗 Vehicle AI Scanner (STABLE VERSION)")

# =============================
# STATE
# =============================
if "run" not in st.session_state:
    st.session_state.run = False

# =============================
# DEVICE
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Using device: {device}")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    if not os.path.exists("best.pt"):
        st.error("❌ best.pt model file not found!")
        st.stop()
    return YOLO("best.pt")

model = load_model()
CLASS_NAMES = model.names

# =============================
# UPLOAD
# =============================
video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start"):
        if video:
            st.session_state.run = True
        else:
            st.warning("Upload video first")

with col2:
    if st.button("🛑 Stop"):
        st.session_state.run = False

if video is None:
    st.stop()

# =============================
# PROCESS
# =============================
if st.session_state.run:

    # Save temp video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    progress = st.progress(0)

    logs = []
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    skip_frames = 5  # 🔥 faster

    while cap.isOpened() and st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        if frame_id % skip_frames != 0:
            continue

        # Resize for speed
        frame = cv2.resize(frame, (640, 360))

        results = model(frame, device=device, verbose=False)[0]

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = CLASS_NAMES[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

                logs.append({
                    "frame": frame_id,
                    "object": label,
                    "confidence": round(conf, 2)
                })

        stframe.image(frame, channels="BGR", use_container_width=True)
        progress.progress(min(frame_id / total_frames, 1.0))

    cap.release()
    os.remove(tfile.name)

    st.success("✅ Done!")

    # =============================
    # LOGS
    # =============================
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df)

        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )
