import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🚗 Vehicle Wire Scanner PRO", layout="wide")
st.title("🔌 Vehicle Wire Continuity Screener (Tuned Version)")

# =============================
# Upload Video
# =============================
video_file = st.file_uploader("Upload Vehicle Inspection Video", type=["mp4", "avi", "mov"])

# =============================
# Detection Function
# =============================
def detect_wire_cut(frame):
    frame = cv2.resize(frame, (640, 360))

    # ROI (focus center area)
    h, w, _ = frame.shape
    frame = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    # Blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Gray
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Normalize brightness
    gray = cv2.equalizeHist(gray)

    # Edge detect
    edges = cv2.Canny(gray, 50, 150)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # =============================
    # Feature 1: Edge Density
    # =============================
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = edge_pixels / total_pixels

    # =============================
    # Feature 2: Contours
    # =============================
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)

    # =============================
    # Feature 3: Lines
    # =============================
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                            minLineLength=50, maxLineGap=10)
    line_count = 0 if lines is None else len(lines)

    # =============================
    # Decision
    # =============================
    if edge_density < 0.01 or line_count < 5:
        return "CUT / DAMAGE", "red"
    elif contour_count > 300:
        return "NOISY / CHECK", "orange"
    else:
        return "NORMAL", "green"


# =============================
# Process Video
# =============================
report_data = []
status_history = []

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

        # ⏩ Skip frames for speed
        if frame_id % 5 != 0:
            frame_id += 1
            continue

        status, color = detect_wire_cut(frame)

        # =============================
        # Stability (last 5 frames)
        # =============================
        status_history.append(status)
        if len(status_history) > 5:
            status_history.pop(0)

        final_status = max(set(status_history), key=status_history.count)

        report_data.append({
            "Frame": frame_id,
            "Status": final_status
        })

        # =============================
        # Display
        # =============================
        cv2.putText(frame, final_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255) if final_status=="CUT / DAMAGE"
                    else (0,255,0) if final_status=="NORMAL"
                    else (0,165,255), 2)

        display.image(frame, channels="BGR")

        frame_id += 1

    cap.release()
    os.remove(temp_video.name)

    # =============================
    # Report
    # =============================
    df = pd.DataFrame(report_data)
    st.dataframe(df)

    total_frames = len(df)
    cut_frames = len(df[df["Status"] == "CUT / DAMAGE"])
    normal_frames = len(df[df["Status"] == "NORMAL"])
    noisy_frames = len(df[df["Status"] == "NOISY / CHECK"])

    damage_pct = (cut_frames / total_frames) * 100 if total_frames > 0 else 0

    st.subheader("📊 Evaluation Report")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"Normal Frames: {normal_frames}")
    st.write(f"Cut/Damage Frames: {cut_frames}")
    st.write(f"Noisy Frames: {noisy_frames}")
    st.write(f"Damage Percentage: {damage_pct:.2f}%")

    # =============================
    # Download Report
    # =============================
    st.download_button(
        "⬇️ Download Report CSV",
        df.to_csv(index=False).encode("utf-8"),
        "wire_report.csv",
        "text/csv"
    )

    # =============================
    # Summary
    # =============================
    st.markdown(f"""
    ### 📋 Final Summary (Telugu + English)

    - మొత్తం ఫ్రేమ్స్: {total_frames}  
    - సాధారణ ఫ్రేమ్స్: {normal_frames}  
    - డ్యామేజ్ ఫ్రేమ్స్: {cut_frames}  
    - నాయిస్ ఫ్రేమ్స్: {noisy_frames}  
    - డ్యామేజ్ శాతం: {damage_pct:.2f}%  

    🚗 Vehicle wire harness shows **{damage_pct:.2f}% damage** in this video.
    """)
