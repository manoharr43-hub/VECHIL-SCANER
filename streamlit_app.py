import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import pandas as pd
import torch
from datetime import datetime

# =============================
# 1. పేజీ సెటప్
# =============================
st.set_page_config(page_title="AI Vehicle Scanner PRO", layout="wide")
st.title("🚗 AI Vehicle Scanner & Damage Tracker")
st.write("సాధారణ మొబైల్ వీడియోను అప్‌లోడ్ చేసి వాహనాలను మరియు డ్యామేజ్‌లను స్కాన్ చేయండి.")

# =============================
# 2. మోడల్ లోడింగ్ (best.pt లేకపోయినా పనిచేస్తుంది)
# =============================
@st.cache_resource
def load_model():
    # best.pt ఉంటే అది తీసుకుంటుంది, లేదంటే yolov8n.pt డౌన్‌లోడ్ చేస్తుంది
    if os.path.exists("best.pt"):
        return YOLO("best.pt")
    else:
        return YOLO("yolov8n.pt") 

model = load_model()
CLASS_NAMES = model.names

# =============================
# 3. సైడ్‌బార్ సెట్టింగ్స్
# =============================
st.sidebar.header("⚙️ Settings")
conf_threshold = st.sidebar.slider("AI Confidence", 0.1, 1.0, 0.3)
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 4. వీడియో అప్‌లోడ్
# =============================
video_file = st.file_uploader("📂 వీడియో ఫైల్‌ను ఇక్కడ అప్‌లోడ్ చేయండి", type=["mp4", "mov", "avi"])

if video_file:
    if st.button("🚀 స్కాన్ ప్రారంభించు (Start Scan)"):
        # టెంపరరీ ఫైల్ సేవింగ్
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        # డేటా సేవింగ్ కోసం
        scan_results = []
        frame_id = 0
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_id += 1
            if frame_id % 5 != 0: # వేగం కోసం ప్రతి 5వ ఫ్రేమ్
                continue

            # AI డిటెక్షన్
            results = model.predict(frame, conf=conf_threshold, device=device, verbose=False)[0]
            
            if len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = CLASS_NAMES[cls_id]
                    conf = float(box.conf[0])
                    
                    # రిపోర్ట్ డేటా
                    scan_results.append({
                        "Frame": frame_id,
                        "Object Detected": label,
                        "Confidence": f"{conf:.2%}",
                        "Time": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # స్క్రీన్ మీద డ్రాయింగ్
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # వీడియో డిస్ప్లే
            stframe.image(frame, channels="BGR", use_container_width=True)
            progress_bar.progress(min(frame_id / total_frames, 1.0))

        cap.release()
        os.remove(tfile.name)

        # =============================
        # 5. ఫైనల్ రిపోర్ట్ జనరేషన్
        # =============================
        st.success("✅ స్కాన్ పూర్తయింది!")
        
        if scan_results:
            st.subheader("📊 Scan Analysis Report")
            df = pd.DataFrame(scan_results)
            
            # సమ్మరీ (ఏవి ఎన్ని ఉన్నాయి)
            summary = df['Object Detected'].value_counts()
            st.bar_chart(summary)
            
            # డేటా టేబుల్
            st.dataframe(df)
            
            # CSV డౌన్‌లోడ్
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 డౌన్‌లోడ్ రిపోర్ట్ (Download CSV)", data=csv, file_name="scan_report.csv")
