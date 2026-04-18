import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🔥 Wire Scanner PRO MAX", layout="wide")
st.title("🚗 Smart Wire Scanner (Damage Location AI)")

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
# FOLDER CREATE
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

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    status = "OUTER OK"
    color = (0, 255, 0)
    damage_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:  # filter noise
            x, y, w, h = cv2.boundingRect(cnt)

            # abnormal small/long shape = possible break
            if w < 100 or h < 50:
                status = "OUTER DAMAGE"
                color = (0, 0, 255)

                cx = x + w // 2
                cy = y + h // 2

                damage_boxes.append((x, y, w, h, cx, cy))

                # draw box
                cv2.rectangle(frame_resized, (x,y), (x+w,y+h), (0,0,255), 2)

                # label
                cv2.putText(frame_resized, "Damage", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

                # center point
                cv2.circle(frame_resized, (cx, cy), 4, (255,0,0), -1)

    # save screenshot if damage
    if status == "OUTER DAMAGE":
        cv2.imwrite(f"damage_frames/frame_{frame_id}.jpg", frame_resized)

    return status, color, frame_resized, damage_boxes

# =============================
# PROCESS VIDEO
# =============================
report_data = []
damage_flags = []

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

        if frame_id % 5 != 0:
            frame_id += 1
            continue

        status, color, output_frame, boxes = detect_wire(frame, frame_id)

        damage_flag = 1 if status == "OUTER DAMAGE" else 0
        damage_flags.append(damage_flag)

        if boxes:
            for (x, y, w, h, cx, cy) in boxes:
                report_data.append({
                    "Frame": frame_id,
                    "Status": status,
                    "X": x,
                    "Y": y,
                    "Width": w,
                    "Height": h,
                    "Center_X": cx,
                    "Center_Y": cy
                })
        else:
            report_data.append({
                "Frame": frame_id,
                "Status": status,
                "X": None,
                "Y": None,
                "Width": None,
                "Height": None,
                "Center_X": None,
                "Center_Y": None
            })

        # ALERT
        if status == "OUTER DAMAGE":
            st.warning(f"⚠️ Damage detected at frame {frame_id}")

        # display
        cv2.putText(output_frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        display.image(output_frame, channels="BGR")

        frame_id += 1

    cap.release()
    os.remove(temp_video.name)

    df = pd.DataFrame(report_data)
    st.dataframe(df)

    # =============================
    # FINAL DECISION
    # =============================
    outer_damage = len(df[df["Status"] == "OUTER DAMAGE"])

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
    st.line_chart(pd.DataFrame({"Damage": damage_flags}))

    # =============================
    # DOWNLOAD
    # =============================
    st.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "wire_report.csv",
        "text/csv"
    )

    # =============================
    # SHOW IMAGES
    # =============================
    st.subheader("📸 Damage Screenshots")

    images = os.listdir("damage_frames")
    for img in images[:5]:
        st.image(f"damage_frames/{img}")

    # =============================
    # SUMMARY
    # =============================
    st.markdown(f"""
    ### 📋 Summary (Telugu + English)

    - బయట డ్యామేజ్ ఫ్రేమ్స్: {outer_damage}  
    - లోపల టెస్ట్: {continuity}  
    - Final Result: {final_status}  

    📍 Damage location coordinates available in report  
    🚗 Smart inspection completed
    """)
