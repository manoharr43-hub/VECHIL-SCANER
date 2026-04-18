import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

st.set_page_config(page_title="🔥 Wire Scanner STABLE", layout="wide")
st.title("🚗 Wire Scanner (Stable Detection Version)")

# =============================
# SIDEBAR
# =============================
st.sidebar.header("🔌 Internal Wire Test")
continuity = st.sidebar.selectbox(
    "Continuity Test Result",
    ["NOT TESTED", "OK", "FAILED"]
)

# =============================
# UPLOAD
# =============================
video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# =============================
# DETECTION FUNCTION
# =============================
def detect_wire(frame):
    frame = cv2.resize(frame, (640, 360))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    edges = cv2.Canny(gray, 50, 150)

    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    edge_density = edge_pixels / total_pixels

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    status = "OUTER OK"
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 🔥 Strong filtering
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)

            ratio = w / h if h != 0 else 0

            # 🔥 Shape + size filter
            if (w < 80 or h < 40) and (0.2 < ratio < 5) and edge_density < 0.02:
                status = "OUTER DAMAGE"
                boxes.append((x, y, w, h))

    return status, boxes, frame

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

    damage_streak = 0  # 🔥 continuous detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % 5 != 0:
            frame_id += 1
            continue

        status, boxes, output_frame = detect_wire(frame)

        # =============================
        # 🔥 STABILITY LOGIC
        # =============================
        if status == "OUTER DAMAGE":
            damage_streak += 1
        else:
            damage_streak = 0

        confirmed = damage_streak >= 3

        final_status = "OUTER DAMAGE" if confirmed else "OUTER OK"

        damage_flag = 1 if confirmed else 0
        damage_flags.append(damage_flag)

        # =============================
        # DRAW BOX ONLY IF CONFIRMED
        # =============================
        if confirmed:
            for (x, y, w, h) in boxes:
                cv2.rectangle(output_frame, (x,y), (x+w,y+h), (0,0,255), 2)

            # alert only once per streak
            if damage_streak == 3:
                st.warning(f"⚠️ Confirmed Damage at frame {frame_id}")

        # display text
        cv2.putText(output_frame, final_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,0,255) if confirmed else (0,255,0), 2)

        display.image(output_frame, channels="BGR")

        report_data.append({
            "Frame": frame_id,
            "Status": final_status
        })

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
        final_status = "🔴 INTERNAL DAMAGE"
    elif outer_damage > 5:
        final_status = "🟠 OUTER DAMAGE"
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
    # SUMMARY
    # =============================
    st.markdown(f"""
    ### 📋 Summary

    - బయట డ్యామేజ్ ఫ్రేమ్స్: {outer_damage}  
    - లోపల టెస్ట్: {continuity}  
    - Final Result: {final_status}  

    ✅ False detection minimized  
    ✅ Stable detection applied
    """)
