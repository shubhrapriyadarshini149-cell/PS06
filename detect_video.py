import cv2
import argparse
import os
from src.detectors import SafetyDetector
from src.ppe_rules import load_rules, evaluate_compliance
from src.temporal import TemporalSmoother

def draw_video_annotations(image, results, detections, hazards):
    for hazard in hazards:
        x1, y1, x2, y2 = hazard.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(image, f"{hazard.cls.upper()} ({hazard.conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
    for res in results:
        p = res["person"]
        x1, y1, x2, y2 = p.bbox
        status = res["smoothed_status"]
        track_id = p.track_id if p.track_id else "?"
        
        if status == "COMPLIANT":
            color = (0, 255, 0)
            label = f"ID:{track_id} COMPLIANT"
        else:
            color = (0, 0, 255)
            label = f"ID:{track_id} VIOLATION: {', '.join(res['missing'])}"
            
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zone", default="default")
    parser.add_argument("--output", default="outputs/processed_video.mp4")
    args = parser.parse_args()

    config = load_rules("config/rules.yaml")
    zone_rules = config["zones"].get(args.zone, config["zones"]["default"])
    thresholds = config.get("confidence_thresholds", {})
    
    ppe_path = "models/ppe_model.pt" if os.path.exists("models/ppe_model.pt") else "yolo11n.pt"
    fire_path = "models/fire_model.pt" if os.path.exists("models/fire_model.pt") else "yolov8n.pt"
    detector = SafetyDetector(ppe_path, fire_path)
    
    smoother = TemporalSmoother(history_size=30, violation_threshold=0.6, compliance_threshold=0.8)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Failed to open video {args.video}")
        exit(1)
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        detections = detector.track(frame)
        results = evaluate_compliance(detections, zone_rules, thresholds)
        
        for res in results:
            p = res["person"]
            is_violation = (res["status"] == "VIOLATION")
            if p.track_id is not None:
                smooth_status = smoother.update(p.track_id, is_violation)
                res["smoothed_status"] = smooth_status
            else:
                res["smoothed_status"] = res["status"]
                
        hazards = [d for d in detections if d.cls in ["fire", "smoke"]]
        
        out_frame = draw_video_annotations(frame, results, detections, hazards)
        out.write(out_frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Saved video to {args.output}")
