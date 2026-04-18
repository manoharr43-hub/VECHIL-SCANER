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
st.title("🚗 Vehicle AI Scanner PRO (FINAL)")

# =============================
# SESSION STATE
# =============================
if "run" not in st.session_state:
    st.session_state.run = False

# =============================
# DEVICE SETUP
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.caption(f"Running on: {device}")

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    try:
        if os.path.exists("best.pt"):
            return YOLO("best.pt")
        else:
            st.warning("⚠️ best.pt not found, using default model")
            return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Model Error: {e}")
        st.stop()

model = load_model()
CLASS_NAMES = model.names

# =============================
# FILE UPLOAD
# =============================
video = st.file_uploader("📂 Upload Video", type=["mp4", "avi", "mov"])

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start"):
        if video:
            st.session_state.run = True
        else:
            st.warning("Upload video first!")

with col2:
    if st.button("🛑 Stop"):
        st.session_state.run = False

if video is None:
    st.info("Upload a video to begin")
    st.stop()

# =============================
# PROCESS VIDEO
# =============================
if st.session_state.run:

    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Cannot open video")
        st.stop()

    stframe = st.empty()
    progress = st.progress(0)

    logs = []
    vehicle_count = {}
    frame_id = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # SPEED SETTINGS
    skip_frames = 6

    st.info("🚀 Processing started...")

    try:
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            # Skip frames
            if frame_id % skip_frames != 0:
                continue

            # Resize for speed
            frame = cv2.resize(frame, (640, 360))

            # YOLO detection
            results = model(frame, device=device, verbose=False)[0]

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Filter low confidence
                    if conf < 0.4:
                        continue

                    label = CLASS_NAMES[cls_id]

                    # Count vehicles
                    if label not in vehicle_count:
                        vehicle_count[label] = 0
                    vehicle_count[label] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw box
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

            # Show video
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Progress bar
            progress.progress(min(frame_id / total_frames, 1.0))

        st.success("✅ Processing Completed")

    except Exception as e:
        st.error(f"❌ Error: {e}")

    finally:
        cap.release()
        os.remove(tfile.name)

    # =============================
    # RESULTS
    # =============================
    if logs:
        df = pd.DataFrame(logs)

        st.subheader("📊 Detection Logs")
        st.dataframe(df)

        # Download CSV
        st.download_button(
            "⬇️ Download Results",
            df.to_csv(index=False),
            "results.csv",
            "text/csv"
        )

    # =============================
    # VEHICLE COUNT DISPLAY
    # =============================
    if vehicle_count:
        st.subheader("🚗 Vehicle Count Summary")

        count_df = pd.DataFrame(
            vehicle_count.items(),
            columns=["Vehicle", "Count"]
        )

        st.dataframe(count_df)

        st.bar_chart(count_df.set_index("Vehicle"))
