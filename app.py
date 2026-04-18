import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Vehicle AI System PRO", layout="wide")
st.title("🚗 Vehicle AI Scanner PRO (FAST MODE)")

# Load model only once (IMPORTANT SPEED FIX)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload video
video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# STOP BUTTON
stop = st.button("🛑 Stop Processing")

if video:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)

    stframe = st.empty()
    progress = st.progress(0)

    logs = []
    frame_id = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # =============================
    # PROCESS VIDEO
    # =============================
    while cap.isOpened():
        if stop:
            st.warning("Processing Stopped by User")
            break

        ret, frame = cap.read()
        if not ret:
            break

        # FAST YOLO INFERENCE
        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        result = results[0]
        output = result.plot()

        # SAFE DETECTION
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.data.cpu().numpy()

            for box in boxes:
                logs.append({
                    "frame": frame_id,
                    "class_id": int(box[5]),
                    "confidence": float(box[4]),
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                })

        # SHOW ONLY EVERY 15th FRAME (FAST UI)
        if frame_id % 15 == 0:
            stframe.image(output, channels="BGR", use_container_width=True)

        frame_id += 1

        # PROGRESS BAR
        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

        # LIMIT (PREVENT FREEZE)
        if frame_id > 300:
            st.warning("Max frame limit reached (FAST MODE)")
            break

    cap.release()

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
        st.warning("No detections found.")
