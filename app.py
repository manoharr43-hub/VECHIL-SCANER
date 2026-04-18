import streamlit as st
import random
import time

st.set_page_config(page_title="🚗 Vehicle Wire Scan Prototype", layout="wide")
st.title("🔌 Vehicle Wire Continuity Screener")

# =============================
# Manual Entry Mode
# =============================
st.sidebar.header("Manual Entry Mode")
wire_id_manual = st.sidebar.text_input("Enter Wire Name (e.g., Engine_Main)")
voltage_manual = st.sidebar.number_input("Enter Voltage (V)", min_value=0.0, max_value=15.0, step=0.1)
resistance_manual = st.sidebar.number_input("Enter Resistance (Ω)", min_value=0.0, max_value=10.0, step=0.1)

# =============================
# Simulation Mode
# =============================
def read_obd_data():
    data = {
        "wire_id": random.choice(["Engine_Main", "Headlight", "Battery_Line", "AC_Wire"]),
        "voltage": round(random.uniform(11.5, 12.5), 2),
        "resistance": round(random.uniform(0.1, 5.0), 2)
    }
    return data

# =============================
# Wire Continuity Check Logic
# =============================
def check_wire_status(voltage, resistance):
    if voltage < 11.0 or resistance > 3.0:
        return "CUT / DAMAGE", "red"
    else:
        return "NORMAL", "green"

# =============================
# Display Section
# =============================
tab1, tab2 = st.tabs(["Simulation Mode", "Manual Entry Mode"])

with tab1:
    st.subheader("🔄 Auto Refresh Simulation")
    placeholder = st.empty()
    if st.button("Start Simulation"):
        while True:
            obd_data = read_obd_data()
            status, color = check_wire_status(obd_data["voltage"], obd_data["resistance"])
            with placeholder.container():
                st.subheader(f"Wire: {obd_data['wire_id']}")
                st.metric("Voltage (V)", obd_data["voltage"])
                st.metric("Resistance (Ω)", obd_data["resistance"])
                st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
            time.sleep(5)

with tab2:
    st.subheader("✍️ Manual Entry Check")
    if wire_id_manual:
        status, color = check_wire_status(voltage_manual, resistance_manual)
        st.subheader(f"Wire: {wire_id_manual}")
        st.metric("Voltage (V)", voltage_manual)
        st.metric("Resistance (Ω)", resistance_manual)
        st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
