import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🚗 Vehicle Wire Video Scanner", layout="wide")
st.title("🔌 Vehicle Wire Continuity Screener (Video + Report)")

# =============================
# Upload Video
# =============================
video_file = st.file_uploader("Upload Vehicle Inspection Video", type=["mp4", "avi", "mov"])

# =============================
# Wire Cut Detection Logic
# =============================
def detect_wire_cut(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_count = np.sum(edges > 0)
    # Threshold heuristic: tune as needed
    if edge_count < 5000:
        return "CUT / DAMAGE", "red"
    else:
        return "NORMAL", "green"

# =============================
# Process Video
# =============================
report_data = []
if video_file is not None:
    # Save uploaded file to temp location
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        status, color = detect_wire_cut(frame)
        report_data.append({"Frame": frame_id, "Status": status})
        frame_id += 1
    cap.release()
    os.remove(temp_video.name)

    # =============================
    # Report Display
    # =============================
    df = pd.DataFrame(report_data)
    st.dataframe(df)

    # Summary
    total_frames = len(df)
    cut_frames = len(df[df["Status"] == "CUT / DAMAGE"])
    normal_frames = total_frames - cut_frames
    damage_pct = (cut_frames / total_frames) * 100 if total_frames > 0 else 0

    st.subheader("📊 Evaluation Report")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"Normal Frames: {normal_frames}")
    st.write(f"Cut/Damage Frames: {cut_frames}")
    st.write(f"Damage Percentage: {damage_pct:.2f}%")

    # =============================
    # Export Options
    # =============================
    st.download_button("⬇️ Export Report (Excel)", 
                       df.to_csv(index=False).encode("utf-8"), 
                       "wire_report.csv", "text/csv")

    # Telugu + English Summary
    st.markdown(f"""
    **Report Summary (Telugu + English):**  
    - మొత్తం ఫ్రేమ్స్: {total_frames}  
    - సాధారణ ఫ్రేమ్స్: {normal_frames}  
    - కట్/డ్యామేజ్ ఫ్రేమ్స్: {cut_frames}  
    - డ్యామేజ్ శాతం: {damage_pct:.2f}%  

    Vehicle wire harness shows {damage_pct:.2f}% damage in scanned video.
    """)
