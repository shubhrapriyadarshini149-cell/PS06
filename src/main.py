import time
import yaml
import cv2
import os
from src.stream import CameraStream
from src.detectors import SafetyDetector
from src.ppe_rules import load_rules, evaluate_compliance
from src.fire_rules import evaluate_fire_hazards
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
    
    ppe_path = "models/ppe_model.pt" if os.path.exists("models/ppe_model.pt") else "yolo11n.pt"
    fire_path = "models/fire_model.pt" if os.path.exists("models/fire_model.pt") else "yolov8n.pt"
    detector = SafetyDetector(ppe_path, fire_path)
    
    alert_mgr = AlertManager()
    
    streams = {}
    smoothers = {}
    
    for c in cameras_config:
        cam_id = c["id"]
        stream = CameraStream(cam_id, c["source"], c.get("fps_limit", 15))
        stream.start()
        streams[cam_id] = stream
        smoothers[cam_id] = TemporalSmoother()
        
    print("Multi-Camera Pipeline Started. Press Ctrl+C to stop.")
    
    try:
        while True:
            for c in cameras_config:
                cam_id = c["id"]
                zone = c["zone"]
                zone_rules = rules_config["zones"].get(zone, rules_config["zones"]["default"])
                
                frame = streams[cam_id].read()
                if frame is None:
                    continue
                    
                detections = detector.track(frame)
                persons = [d for d in detections if d.cls == "person" and d.conf >= thresholds.get("person", 0.5)]
                
                ppe_results = evaluate_compliance(detections, zone_rules, thresholds)
                fire_events = evaluate_fire_hazards(detections, persons)
                
                for res in ppe_results:
                    p = res["person"]
                    is_violation = (res["status"] == "VIOLATION")
                    if p.track_id is not None:
                        smooth_status = smoothers[cam_id].update(p.track_id, is_violation)
                        if smooth_status == "VIOLATION":
                            details = f"Missing PPE: {', '.join(res['missing'])}"
                            alert_mgr.trigger_alert(cam_id, zone, "PPE_VIOLATION", p.track_id, "HIGH", details, frame)
                            
                for event in fire_events:
                    alert_mgr.trigger_alert(cam_id, zone, "FIRE_HAZARD", 0, event["severity"], event["details"], frame)
                    
            time.sleep(0.05) 
    except KeyboardInterrupt:
        print("Stopping streams...")
        for s in streams.values():
            s.stop()

if __name__ == "__main__":
    main()
