import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd

st.set_page_config(page_title="Vehicle AI System", layout="wide")
st.title("🚗 Vehicle AI Scanner")

model = YOLO("best.pt")

video = st.file_uploader("Upload Video")

if video:

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video.read())

    cap = cv2.VideoCapture(temp.name)

    stframe = st.empty()

    logs = []
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        output = results[0].plot()

        for box in results[0].boxes.data:
            logs.append({
                "frame": frame_id,
                "class": int(box[5]),
                "confidence": float(box[4])
            })

        stframe.image(output, channels="BGR")

        frame_id += 1

    cap.release()

    df = pd.DataFrame(logs)

    st.subheader("📊 Report")
    st.dataframe(df)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "report.csv",
        "text/csv"
    )
