import streamlit as st

st.set_page_config(page_title="🚗 Vehicle Wire Scan Prototype", layout="wide")
st.title("🔌 Vehicle Wire Continuity Screener")

# =============================
# Text + Number Inputs
# =============================
wire_id = st.text_input("Enter Wire Name (e.g., Engine_Main)")
voltage = st.number_input("Enter Voltage (V)", min_value=0.0, max_value=15.0, step=0.1)
resistance = st.number_input("Enter Resistance (Ω)", min_value=0.0, max_value=10.0, step=0.1)

# =============================
# Wire Continuity Check Logic
# =============================
def check_wire_status(voltage, resistance):
    if voltage < 11.0 or resistance > 3.0:
        return "CUT / DAMAGE", "red"
    else:
        return "NORMAL", "green"

# =============================
# Display Result
# =============================
if wire_id:
    status, color = check_wire_status(voltage, resistance)
    st.subheader(f"Wire: {wire_id}")
    st.metric("Voltage (V)", voltage)
    st.metric("Resistance (Ω)", resistance)
    st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)
