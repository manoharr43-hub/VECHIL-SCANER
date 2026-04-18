import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd

st.set_page_config(page_title="Vehicle AI System", layout="wide")
st.title("🚗 Vehicle AI Scanner (PRO)")

# Load model once
model = YOLO("best.pt")

video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        result = results[0]
        output = result.plot()

        # SAFE detection handling
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

        # Show frame every 5th frame (faster UI)
        if frame_id % 5 == 0:
            stframe.image(output, channels="BGR", use_container_width=True)

        frame_id += 1

        # progress bar update
        if total_frames > 0:
            progress.progress(min(frame_id / total_frames, 1.0))

    cap.release()

    # DataFrame
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
