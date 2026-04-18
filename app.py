pip install streamlit pyserial
streamlit run wire_scan_app.py
import streamlit as st
import random
import time

st.set_page_config(page_title="🚗 Vehicle Wire Scan Prototype", layout="wide")
st.title("🔌 Vehicle Wire Continuity Screener")

def read_obd_data():
    data = {
        "wire_id": random.choice(["Engine_Main", "Headlight", "Battery_Line", "AC_Wire"]),
        "voltage": round(random.uniform(11.5, 12.5), 2),
        "resistance": round(random.uniform(0.1, 5.0), 2)
    }
    return data

def check_wire_status(voltage, resistance):
    if voltage < 11.0 or resistance > 3.0:
        return "CUT / DAMAGE", "red"
    else:
        return "NORMAL", "green"

placeholder = st.empty()

while True:
    obd_data = read_obd_data()
    status, color = check_wire_status(obd_data["voltage"], obd_data["resistance"])
    with placeholder.container():
        st.subheader(f"Wire: {obd_data['wire_id']}")
        st.metric("Voltage (V)", obd_data["voltage"])
        st.metric("Resistance (Ω)", obd_data["resistance"])
        st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
    time.sleep(5)
