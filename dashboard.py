import streamlit as st
import sqlite3
import pandas as pd
import os

st.set_page_config(page_title="Factory Safety Dashboard", layout="wide")

DB_PATH = "outputs/logs.db"

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM incident_logs ORDER BY id DESC LIMIT 50", conn)
    return df

st.title("🛡️ AI Factory Safety Dashboard")

df = load_data()

if df.empty:
    st.info("No safety incidents logged yet. Start the pipeline using src/main.py!")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Incidents (Last 50)", len(df))
    col2.metric("Critical Hazards", len(df[df["severity"] == "CRITICAL"]))
    col3.metric("PPE Violations", len(df[df["incident_type"] == "PPE_VIOLATION"]))

    st.subheader("Recent Safety Alerts")
    
    for idx, row in df.head(10).iterrows():
        with st.expander(f"{row['timestamp']} | {row['zone']} | {row['severity']} - {row['incident_type']}"):
            st.write(f"**Camera:** {row['camera_id']} (Track ID: {row['track_id']})")
            st.write(f"**Details:** {row['details']}")
            if os.path.exists(row["snapshot_path"]):
                st.image(row["snapshot_path"], caption=row['incident_type'], use_container_width=True)
            else:
                st.warning("Snapshot image not found.")
                
    st.subheader("Incident Data Log")
    st.dataframe(df.drop(columns=["snapshot_path"]))
