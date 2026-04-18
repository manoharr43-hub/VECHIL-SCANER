import streamlit as st
import cv2
import tempfile
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="🚗 Vehicle AI Scanner", layout="wide")
st.title("🔥 Vehicle Inner-Part AI System")

st.markdown("Upload vehicle video for AI detection")

# =========================
# SAFE YOLO IMPORT
# =========================
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # safe fallback model
    yolo_enabled = True
except:
    st.warning("YOLO not loaded, running in SAFE MODE")
    yolo_enabled = False

# =========================
# UPLOAD VIDEO
# =========================
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# =========================
# PROCESS VIDEO
# =========================
if video_file:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    logs = []
    frame_id = 0
    damage_score = 0

    st.success("Processing Started 🚀")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # =========================
        # YOLO DETECTION
        # =========================
        if yolo_enabled:
            results = model(frame)
            output = results[0].plot()

            # LOG DATA
            if results[0].boxes is not None:
                for box in results[0].boxes.data:
                    x1, y1, x2, y2, conf, cls = box

                    logs.append({
                        "frame": frame_id,
                        "class": int(cls),
                        "confidence": float(conf)
                    })

                    # Example damage detection
                    if int(cls) == 5:
                        damage_score += 1
        else:
            output = frame  # SAFE MODE (no AI)

        # =========================
        # SHOW FRAME
        # =========================
        stframe.image(output, channels="BGR", use_container_width=True)

        frame_id += 1

    cap.release()

    # =========================
    # REPORT
    # =========================
    st.subheader("📊 Detection Report")

    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df)

        st.download_button(
            "⬇️ Download Report",
            df.to_csv(index=False).encode(),
            "vehicle_report.csv",
            "text/csv"
        )
    else:
        st.info("No detection logs (SAFE MODE or no objects detected)")

    # =========================
    # FINAL STATUS
    # =========================
    st.subheader("📌 Final Result")

    if damage_score > 10:
        status = "🔴 HEAVY DAMAGE"
    elif damage_score > 0:
        status = "🟠 MINOR DAMAGE"
    else:
        status = "🟢 NORMAL / SAFE"

    st.write("Damage Score:", damage_score)
    st.write("Status:", status)

    st.success("Processing Completed 🚀")
