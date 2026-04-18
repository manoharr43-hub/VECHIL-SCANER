import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import pandas as pd

st.set_page_config(page_title="Vehicle AI System", layout="wide")
st.title("🚗 Vehicle AI Scanner")

model = YOLO("best.pt")

video = st.file_uploader("Upload Video", type=["mp4","avi","mov"])

if video:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(video.read())
    temp.close()

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

        # Convert tensor to numpy
        for box in results[0].boxes.data.cpu().numpy():
            logs.append({
                "frame": frame_id,
                "class": int(box[5]),
                "confidence": float(box[4])
            })

        # Show every 10th frame only
        if frame_id % 10 == 0:
            stframe.image(output, channels="BGR")

        frame_id += 1

    cap.release()

    df = pd.DataFrame(logs)

    st.subheader("📊 Detection Report")
    st.dataframe(df)

    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode(),
        "report.csv",
        "text/csv"
    )
