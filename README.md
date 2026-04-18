
# VECHIL-SCANER

🔌 Vehicle Wire Continuity Screener App  
Detects wire cut / damage in vehicles using OBD-II data and AI logic.

---

## 🚀 Features
- OBD-II data reader (Voltage, Resistance)
- Wire continuity check logic
- Color-coded status (Green = Normal, Red = Cut/Damage)
- Streamlit dashboard UI
- Expandable for sector dropdown (Engine, Lights, Battery, AC)

---

## 🛠️ Installation
```bash
git clone https://github.com/manoharr43-hub/VECHIL-SCANER.git
cd VECHIL-SCANER
pip install streamlit pyserial
streamlit run wire_scan_app.py


