import sqlite3
import time
import os
import cv2
import requests

class AlertManager:
    def __init__(self, db_path="outputs/logs.db", telegram_token="", chat_id="", webhook_url=""):
        self.db_path = db_path
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        self.webhook_url = webhook_url
        self.cooldown_cache = {} # (cam_id, track_id, type) -> timestamp
        self._init_db()
        
    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS incident_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT,
                            camera_id TEXT,
                            zone TEXT,
                            incident_type TEXT,
                            track_id INTEGER,
                            severity TEXT,
                            details TEXT,
                            snapshot_path TEXT
                        )''')
                        
    def trigger_alert(self, cam_id, zone, incident_type, track_id, severity, details, frame):
        cache_key = (cam_id, track_id, incident_type)
        now = time.time()
        
        # 60s cooldown unless it escalates to CRITICAL
        if cache_key in self.cooldown_cache:
            if now - self.cooldown_cache[cache_key] < 60 and severity != "CRITICAL":
                return 
                
        self.cooldown_cache[cache_key] = now
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        snap_dir = "outputs/snapshots"
        os.makedirs(snap_dir, exist_ok=True)
        snap_path = f"{snap_dir}/{timestamp}_{cam_id}_{incident_type}.jpg"
        cv2.imwrite(snap_path, frame)
        
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO incident_logs (timestamp, camera_id, zone, incident_type, track_id, severity, details, snapshot_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (ts_str, cam_id, zone, incident_type, track_id, severity, details, snap_path))
                         
        print(f"[{ts_str}] ALERT [{severity}]: {incident_type} in {zone} (Cam: {cam_id}). {details}")
        
        msg = f"🚨 {severity} ALERT\nLocation: {zone} (Cam: {cam_id})\nEvent: {incident_type}\nDetails: {details}"
        
        if self.telegram_token and self.chat_id:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendPhoto"
            try:
                with open(snap_path, 'rb') as f:
                    requests.post(url, data={'chat_id': self.chat_id, 'caption': msg}, files={'photo': f})
            except Exception as e:
                print(f"Failed to send Telegram alert: {e}")
                
        if self.webhook_url:
            try:
                payload = {
                    "text": msg,
                    "severity": severity,
                    "zone": zone,
                    "camera_id": cam_id,
                    "incident_type": incident_type
                }
                requests.post(self.webhook_url, json=payload)
            except Exception as e:
                print(f"Failed to send Webhook alert: {e}")
