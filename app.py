import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd
import numpy as np

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="🚗 Vehicle AI Scanner", layout="wide")
st.title("🔥 REAL Vehicle Inner-Part AI System")

st.markdown("Upload video → detect engine parts, wires, ECU, battery, damage")

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "best.pt"   # trained YOLO model
model = YOLO(MODEL_PATH)

# =========================
# UPLOAD VIDEO
# =========================
video_file = st.file_uploader("Upload Vehicle Video", type=["mp4","avi","mov"])

# =========================
# PROCESS
# =========================
if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    log_data = []
    frame_id = 0

    st.success("AI Processing Started 🚀")

    damage_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction
        results = model(frame)

        annotated = results[0].plot()

        # =========================
        # LOGGING DETECTIONS
        # =========================
        if results[0].boxes is not None:
            for box in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = box
                log_data.append({
                    "frame": frame_id,
                    "class": int(cls),
                    "confidence": float(conf)
                })

                # damage class example (class 5 = damaged_wire)
                if int(cls) == 5:
                    damage_count += 1

        # =========================
        # SHOW VIDEO
        # =========================
        stframe.image(annotated, channels="BGR", use_container_width=True)

        frame_id += 1

    cap.release()

    # =========================
    # REPORT
    # =========================
    df = pd.DataFrame(log_data)

    st.subheader("📊 Detection Report")
    st.dataframe(df)

    # =========================
    # FINAL RESULT
    # =========================
    st.subheader("📌 Final Status")

    if damage_count > 10:
        status = "🔴 HEAVY DAMAGE DETECTED"
    elif damage_count > 0:
        status = "🟠 MINOR DAMAGE DETECTED"
    else:
        status = "🟢 NO DAMAGE FOUND"

    st.write("Damage Score:", damage_count)
    st.write("Status:", status)

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.download_button(
        "⬇️ Download CSV Report",
        df.to_csv(index=False).encode(),
        "vehicle_ai_report.csv",
        "text/csv"
    )

    st.success("Processing Completed 🚀")
