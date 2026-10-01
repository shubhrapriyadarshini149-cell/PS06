import time
import yaml
import cv2
import os
from src.stream import CameraStream
from src.detectors import SafetyDetector
from src.ppe_rules import load_rules, evaluate_compliance
from src.fire_rules import evaluate_fire_hazards, FireTemporalSmoother
from src.temporal import TemporalSmoother
from src.alerts import AlertManager

def load_cameras(path="config/cameras.yaml"):
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return yaml.safe_load(f).get("cameras", [])

def main():
    cameras_config = load_cameras()
    if not cameras_config:
        print("No cameras configured in config/cameras.yaml")
        return
        
    rules_config = load_rules("config/rules.yaml")
    thresholds = rules_config.get("confidence_thresholds", {})
    
    ppe_path = "models/ppe.pt" if os.path.exists("models/ppe.pt") else "models/ppe_model.pt"
    if not os.path.exists(ppe_path): ppe_path = "yolo11n.pt"
    fire_path = "models/fire.pt" if os.path.exists("models/fire.pt") else "models/fire_model.pt"
    if not os.path.exists(fire_path): fire_path = "yolov8n.pt"
    
    print("Loading AI Models...")
    detector = SafetyDetector(ppe_path, fire_path)
    
    # Check for webhook or tokens in env for alerts
    webhook = os.getenv("SLACK_WEBHOOK_URL", "")
    alert_mgr = AlertManager(webhook_url=webhook)
    
    streams = {}
    smoothers = {}
    fire_smoothers = {}
    frame_counts = {}
    
    for c in cameras_config:
        cam_id = c["id"]
        stream = CameraStream(cam_id, c["source"], c.get("fps_limit", 15))
        stream.start()
        streams[cam_id] = stream
        smoothers[cam_id] = TemporalSmoother(n_frames=15, m_frames=10, max_age=30)
        fire_smoothers[cam_id] = FireTemporalSmoother(persistence_frames=5, escalation_frames=10)
        frame_counts[cam_id] = 0
        
    print("Multi-Camera Live RTSP Pipeline Started. Press Ctrl+C to stop.")
    
    try:
        while True:
            for c in cameras_config:
                cam_id = c["id"]
                zone = c["zone"]
                zone_rules = rules_config["zones"].get(zone, rules_config["zones"]["default"])
                
                frame = streams[cam_id].read()
                if frame is None:
                    continue
                    
                frame_counts[cam_id] += 1
                fc = frame_counts[cam_id]
                
                # 1. Track Persons & PPE
                detections = detector.track(frame)
                persons = [d for d in detections if d.cls == "person" and d.conf >= thresholds.get("person", 0.5)]
                ppe_results = evaluate_compliance(detections, zone_rules, thresholds)
                
                for res in ppe_results:
                    p = res["person"]
                    if p.track_id is not None:
                        smoothed_missing = smoothers[cam_id].update(p.track_id, res["missing"], fc)
                        if smoothed_missing:
                            details = f"Missing PPE: {', '.join(smoothed_missing)}"
                            alert_mgr.trigger_alert(cam_id, zone, "PPE_VIOLATION", p.track_id, "HIGH", details, frame)
                
                # Clean old PPE tracks
                smoothers[cam_id].cleanup(fc)
                
                # 2. Fire & Smoke
                fire_detections = detector.detect_fire(frame)
                fire_events = evaluate_fire_hazards(fire_detections, persons, fc, fire_smoothers[cam_id])
                
                for event in fire_events:
                    # To avoid duplicate alerts for the same continuous fire, AlertManager handles cooldowns
                    alert_mgr.trigger_alert(cam_id, zone, "FIRE_HAZARD", 0, event["severity"], event["details"], frame)
                    
            time.sleep(0.05) 
    except KeyboardInterrupt:
        print("Stopping streams...")
        for s in streams.values():
            s.stop()

if __name__ == "__main__":
    main()
