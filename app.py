import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Vehicle AI Scanner PRO", layout="wide")
st.title("🚗 Vehicle AI Scanner (STABLE PRO VERSION)")

# =============================
# SESSION STATE CONTROL
# =============================
if "run" not in st.session_state:
    st.session_state.run = False

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# =============================
# UPLOAD VIDEO
# =============================
video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

col1, col2 = st.columns(2)

with col1:
    start = st.button("▶️ Start Processing")

with col2:
    stop = st.button("🛑 Stop")

if start:
    st.session_state.run = True

if stop:
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

    st.info("Processing started... 🚀")

    while cap.isOpened() and st.session_state.run:

        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 SKIP FRAMES FOR SPEED
        if frame_id % 2 != 0:
            frame_id += 1
            continue

        results = model(frame, imgsz=512, conf=0.4, verbose=False)
        result = results[0]
        output = result.plot()

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

        # UI update every 5 frames
        if frame_id % 5 == 0:
            stframe.image(output, channels="BGR", use_container_width=True)

        frame_id += 1

        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

        # Safety limit
        if frame_id > 500:
            st.warning("Frame limit reached (SAFE MODE)")
            break

    cap.release()
    st.success("Processing completed ✅")

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
        st.warning("No objects detected in video.")
