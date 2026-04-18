import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🔌 Smart Wire Scanner PRO", layout="wide")
st.title("🚗 Vehicle Wire Scanner (Camera + Internal Test)")

# =============================
# USER INPUT (INTERNAL TEST)
# =============================
st.sidebar.header("🔌 Internal Wire Test")

continuity = st.sidebar.selectbox(
    "Continuity Test Result",
    ["NOT TESTED", "OK", "FAILED"]
)

# =============================
# Upload Video
# =============================
video_file = st.file_uploader("Upload Vehicle Video", type=["mp4", "avi", "mov"])

# =============================
# Detection Function (Outside)
# =============================
def detect_wire(frame):
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    edges = cv2.Canny(gray, 50, 150)

    edge_count = np.sum(edges > 0)

    if edge_count < 4000:
        return "OUTER DAMAGE", "red"
    else:
        return "OUTER OK", "green"

# =============================
# PROCESS VIDEO
# =============================
report_data = []

if video_file is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    frame_id = 0
    display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # speed
        if frame_id % 5 != 0:
            frame_id += 1
            continue

        status, color = detect_wire(frame)

        report_data.append({
            "Frame": frame_id,
            "Outer Status": status
        })

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255) if color=="red" else (0,255,0), 2)

        display.image(frame, channels="BGR")

        frame_id += 1

    cap.release()
    os.remove(temp_video.name)

    df = pd.DataFrame(report_data)
    st.dataframe(df)

    # =============================
    # FINAL COMBINED LOGIC
    # =============================
    outer_damage = len(df[df["Outer Status"] == "OUTER DAMAGE"])

    if continuity == "FAILED":
        final_status = "🔴 INTERNAL WIRE DAMAGE"
    elif outer_damage > 5:
        final_status = "🟠 OUTER DAMAGE DETECTED"
    elif continuity == "OK":
        final_status = "🟢 WIRE FULLY OK"
    else:
        final_status = "⚠️ TEST NOT COMPLETE"

    # =============================
    # REPORT
    # =============================
    st.subheader("📊 Final Report")

    st.write(f"Outer Damage Frames: {outer_damage}")
    st.write(f"Internal Test: {continuity}")
    st.write(f"Final Status: {final_status}")

    # =============================
    # DOWNLOAD
    # =============================
    st.download_button(
        "⬇️ Download Report",
        df.to_csv(index=False).encode("utf-8"),
        "wire_report.csv",
        "text/csv"
    )

    # =============================
    # SUMMARY
    # =============================
    st.markdown(f"""
    ### 📋 Summary (Telugu + English)

    - బయట వైర్ పరిస్థితి: {outer_damage} frames damage  
    - లోపల టెస్ట్: {continuity}  
    - Final Result: {final_status}  

    👉 Combined analysis completed.
    """)
