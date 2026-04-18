import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🔥 Wire Scanner ULTIMATE", layout="wide")
st.title("🚗 Smart Wire Scanner (ALL-IN-ONE PRO)")

# =============================
# SIDEBAR - INTERNAL TEST
# =============================
st.sidebar.header("🔌 Internal Wire Test")

continuity = st.sidebar.selectbox(
    "Continuity Test Result",
    ["NOT TESTED", "OK", "FAILED"]
)

# =============================
# UPLOAD VIDEO
# =============================
video_file = st.file_uploader("Upload Inspection Video", type=["mp4", "avi", "mov"])

# =============================
# CREATE SCREENSHOT FOLDER
# =============================
if not os.path.exists("damage_frames"):
    os.makedirs("damage_frames")

# =============================
# DETECTION FUNCTION
# =============================
def detect_wire(frame, frame_id):
    frame_resized = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    edges = cv2.Canny(gray, 50, 150)

    edge_count = np.sum(edges > 0)

    # Find contours (for location)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Default
    status = "OUTER OK"
    color = (0, 255, 0)

    if edge_count < 4000:
        status = "OUTER DAMAGE"
        color = (0, 0, 255)

        # 📍 Draw bounding box
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > 500:  # filter small noise
                cv2.rectangle(frame_resized, (x,y), (x+w,y+h), (0,0,255), 2)

        # 📸 Save screenshot
        cv2.imwrite(f"damage_frames/frame_{frame_id}.jpg", frame_resized)

    return status, color, frame_resized

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

    damage_flag_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 5 != 0:
            frame_id += 1
            continue

        status, color, output_frame = detect_wire(frame, frame_id)

        damage_flag = 1 if status == "OUTER DAMAGE" else 0
        damage_flag_list.append(damage_flag)

        report_data.append({
            "Frame": frame_id,
            "Outer Status": status
        })

        # 🚨 ALERT
        if status == "OUTER DAMAGE":
            st.warning(f"⚠️ Damage detected at frame {frame_id}")

        # Display
        cv2.putText(output_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 2)

        display.image(output_frame, channels="BGR")

        frame_id += 1

    cap.release()
    os.remove(temp_video.name)

    df = pd.DataFrame(report_data)
    st.dataframe(df)

    # =============================
    # FINAL LOGIC
    # =============================
    outer_damage = len(df[df["Outer Status"] == "OUTER DAMAGE"])

    if continuity == "FAILED":
        final_status = "🔴 INTERNAL DAMAGE (CRITICAL)"
    elif outer_damage > 5:
        final_status = "🟠 OUTER DAMAGE DETECTED"
    elif continuity == "OK":
        final_status = "🟢 FULLY OK"
    else:
        final_status = "⚠️ TEST INCOMPLETE"

    # =============================
    # REPORT
    # =============================
    st.subheader("📊 Final Report")

    st.write(f"Outer Damage Frames: {outer_damage}")
    st.write(f"Internal Test: {continuity}")
    st.write(f"Final Status: {final_status}")

    # =============================
    # GRAPH
    # =============================
    st.subheader("📈 Damage Timeline")
    chart_df = pd.DataFrame({"Damage": damage_flag_list})
    st.line_chart(chart_df)

    # =============================
    # DOWNLOAD CSV
    # =============================
    st.download_button(
        "⬇️ Download Report CSV",
        df.to_csv(index=False).encode("utf-8"),
        "wire_report.csv",
        "text/csv"
    )

    # =============================
    # SHOW SAVED IMAGES
    # =============================
    st.subheader("📸 Damage Screenshots")

    image_files = os.listdir("damage_frames")
    for img in image_files[:5]:  # show first 5
        st.image(f"damage_frames/{img}")

    # =============================
    # SUMMARY
    # =============================
    st.markdown(f"""
    ### 📋 Final Summary (Telugu + English)

    - బయట డ్యామేజ్ ఫ్రేమ్స్: {outer_damage}  
    - లోపల టెస్ట్: {continuity}  
    - Final Result: {final_status}  

    🚗 Complete smart analysis done.
    """)
