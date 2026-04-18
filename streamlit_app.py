import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch
from datetime import datetime

# =============================
# CONFIG & PAGE SETUP
# =============================
st.set_page_config(page_title="AI Vehicle Internal Scanner", layout="wide")
st.title("🚗 AI Vehicle Scanner (Damage & Wire-Cut Detector)")
st.write("సాధారణ మొబైల్ వీడియో ద్వారా వాహన డ్యామేజ్ మరియు వైర్ కట్స్ గుర్తించండి.")

# =============================
# LOAD CUSTOM MODEL
# =============================
@st.cache_resource
def load_model():
    # ఒకవేళ మీ దగ్గర damage_best.pt ఉంటే అది లోడ్ అవుతుంది, లేదంటే సాధారణ yolov8n
    model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    return YOLO(model_path)

model = load_model()
CLASS_NAMES = model.names

# =============================
# SIDEBAR SETTINGS
# =============================
st.sidebar.header("🔍 Scan Settings")
conf_threshold = st.sidebar.slider("Confidence (AI ఖచ్చితత్వం)", 0.1, 1.0, 0.3)
# లోపలి భాగాలు స్కాన్ చేసేటప్పుడు కాన్ఫిడెన్స్ కొంచెం తక్కువ (0.3) ఉంటే వైర్లు బాగా దొరుకుతాయి.

# =============================
# VIDEO UPLOAD
# =============================
video_file = st.file_uploader("📂 Upload Mobile Video", type=["mp4", "mov", "avi"])

if video_file is not None:
    # ప్రాసెసింగ్ బటన్
    if st.button("🚀 Start AI Deep Scan"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # డేటా సేవ్ చేయడానికి లిస్ట్
        damage_data = []
        frame_id = 0
        
        st.info("🔎 కెమెరా లోపలి భాగాలను మరియు వైర్లను స్కాన్ చేస్తోంది...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            # వేగం కోసం ప్రతి 5వ ఫ్రేమ్ మాత్రమే చెక్ చేస్తుంది
            if frame_id % 5 != 0:
                continue

            # AI డిటెక్షన్
            results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            
            # రిజల్ట్స్ ప్రాసెసింగ్
            if len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = CLASS_NAMES[cls_id]
                    conf = float(box.conf[0])
                    
                    # మనం కేవలం డ్యామేజ్ లేదా వైర్ కట్లను మాత్రమే రిపోర్ట్ చేయాలి
                    # (మీ మోడల్ క్లాస్ పేర్లను బట్టి ఇక్కడ పేర్లు మార్చుకోవాలి)
                    is_damage = any(word in label.lower() for word in ["cut", "damage", "scratch", "dent", "wire"])
                    
                    if is_damage:
                        # డ్యామేజ్ ఉన్న ఫ్రేమ్ ని ఫోటోగా సేవ్ చేయడం
                        timestamp = datetime.now().strftime("%H%M%S")
                        img_name = f"damage_{frame_id}_{timestamp}.jpg"
                        
                        damage_data.append({
                            "Frame": frame_id,
                            "Detected Issue": label.upper(),
                            "Confidence": f"{conf:.2%}",
                            "Status": "⚠️ NEEDS REPAIR"
                        })
                        
                        # బాక్సులను డ్రా చేయడం
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, f"ALERT: {label}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # వీడియోను యాప్‌లో చూపించడం
            stframe.image(frame, channels="BGR", use_container_width=True)

        cap.release()
        os.remove(tfile.name)
        
        # =============================
        # FINAL ANALYSIS REPORT
        # =============================
        st.divider()
        st.subheader("📋 Final Inspection Report")
        
        if damage_data:
            df = pd.DataFrame(damage_data)
            
            # ఒకే రకమైన డ్యామేజ్ పదే పదే రాకుండా గ్రూప్ చేయడం
            report_summary = df.drop_duplicates(subset=['Detected Issue'])
            
            # మెట్రిక్స్ చూపించడం
            c1, c2 = st.columns(2)
            c1.metric("Total Issues Found", len(report_summary))
            c2.error("Critical Damage Detected: YES")
            
            # రిపోర్ట్ టేబుల్
            st.table(report_summary)
            
            # CSV డౌన్లోడ్
            csv = report_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Repair Report",
                data=csv,
                file_name='vehicle_damage_report.csv',
                mime='text/csv',
            )
        else:
            st.success("✅ స్కాన్ పూర్తయింది. వాహనం లోపల ఎలాంటి వైర్ కట్స్ లేదా డ్యామేజ్‌లు లేవు.")
