import streamlit as st
import sqlite3
import pandas as pd
import os
import json
import subprocess
import time

st.set_page_config(
    page_title="AI Factory Safety Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .badge-c {
        background-color: #10B981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold;
    }
    .badge-nc {
        background-color: #EF4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = "outputs/logs.db"
SUMMARY_PATH = "outputs/demo_summary.json"

st.title("🛡️ AI FACTORY SAFETY MONITORING")
st.markdown("### Safety Gear Compliance System")

# --- System Status Sidebar ---
st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("🟢 **Model:** Loaded (PyTorch)")
st.sidebar.markdown("🟢 **Video:** Processing Engine Active")
st.sidebar.markdown("🟢 **Tracking:** ByteTrack Active")
st.sidebar.markdown("🟢 **Compliance Engine:** Active")
st.sidebar.markdown("🟢 **Database:** Connected (SQLite)")

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 How AI Works")
st.sidebar.markdown("""
1. **Video Input**
2. ↓ **Person Detection**
3. ↓ **PPE Detection**
4. ↓ **Person-PPE Association** (Anatomical Region)
5. ↓ **Compliance Check** (Temporal M-of-N)
6. ↓ **C / NC Decision**
7. ↓ **Safety Alert & Database Log**
""")

# --- Main Flow ---
st.header("1. Upload & Process Video")
uploaded_file = st.file_uploader("Upload 5-6 Second Demo Video (MP4)", type=["mp4"])

if uploaded_file is not None:
    # Save the file
    os.makedirs("tests/videos", exist_ok=True)
    video_path = os.path.join("tests/videos", "uploaded_demo.mp4")
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("Process Video"):
        with st.spinner("Processing Video with AI Pipeline..."):
            # Run detect_video.py via subprocess
            out_mp4 = "outputs/demo_annotated.mp4"
            cmd = ["python", "detect_video.py", video_path, "--output", out_mp4, "--frame-skip", "1"]
            subprocess.run(cmd, check=True)
            
            st.success("Video Processed Successfully!")
            
        # Display Video
        st.header("2. Annotated Output Video")
        if os.path.exists("outputs/demo_annotated.mp4"):
            st.video("outputs/demo_annotated.mp4")
            
        # Load Summary JSON
        if os.path.exists(SUMMARY_PATH):
            with open(SUMMARY_PATH, "r") as f:
                demo_summary = json.load(f)
                
            total_workers = len(demo_summary)
            compliant_workers = sum(1 for w in demo_summary.values() if "C - COMPLIANT" in w["status"])
            non_compliant_workers = total_workers - compliant_workers
            
            st.header("3. Compliance Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("TOTAL WORKERS", total_workers)
            c2.metric("COMPLIANT (C)", compliant_workers)
            c3.metric("NON-COMPLIANT (NC)", non_compliant_workers)
            c4.metric("TOTAL VIOLATIONS", non_compliant_workers)
            
            st.header("4. Worker-by-Worker Breakdown")
            
            # Build Table
            table_data = []
            for track_id, data in demo_summary.items():
                status_raw = data["status"]
                is_nc = "NC" in status_raw
                status_badge = "NC" if is_nc else "C"
                
                row = {
                    "Worker ID": f"#{track_id}",
                    "Status": status_badge,
                }
                
                # Dynamically add supported PPE items
                for item, item_status in data.get("ppe", {}).items():
                    row[item.capitalize()] = "❌ MISSING" if item_status == "MISSING" else "✅ PRESENT"
                    
                if is_nc and data.get("missing"):
                    row["Missing PPE"] = f"Missing: {', '.join(data['missing']).title()}"
                else:
                    row["Missing PPE"] = "None"
                    
                table_data.append(row)
                
            if table_data:
                df_workers = pd.DataFrame(table_data)
                st.table(df_workers)
                
        # Load Alerts
        st.header("5. Safety Alerts Log")
        if os.path.exists(DB_PATH):
            with sqlite3.connect(DB_PATH) as conn:
                df_alerts = pd.read_sql_query("SELECT timestamp, incident_type, track_id, camera_id, severity, details FROM incident_logs ORDER BY id DESC LIMIT 10", conn)
            
            if not df_alerts.empty:
                # Rename columns for display
                df_alerts.columns = ["Time", "Violation", "Worker ID", "Camera Source", "Severity", "Details"]
                st.dataframe(df_alerts, use_container_width=True, hide_index=True)
            else:
                st.info("No safety alerts recorded.")
