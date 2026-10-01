import cv2
import argparse
import os
import time
import torch
import json
from src.detectors import SafetyDetector
from src.ppe_rules import load_rules, evaluate_compliance
from src.temporal import TemporalSmoother
from src.alerts import AlertManager

def draw_video_annotations(image, results, detections, hazards, current_fps):
    for hazard in hazards:
        x1, y1, x2, y2 = hazard["bbox"]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(image, f"{hazard['cls'].upper()} ({hazard['conf']:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
    gear_colors = {
        "hardhat": (255, 255, 0),
        "vest": (0, 255, 255),
        "gloves": (255, 0, 255),
        "boots": (128, 0, 128)
    }
    for d in detections:
        if d.cls in gear_colors:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), gear_colors[d.cls], 1)
            cv2.putText(image, f"{d.cls} {d.conf:.2f}", (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, gear_colors[d.cls], 1)

    for res in results:
        p = res["person"]
        x1, y1, x2, y2 = p.bbox
        status = res["smoothed_status"]
        track_id = p.track_id if p.track_id else "?"
        
        lines = []
        lines.append(f"PERSON #{track_id}")
        lines.append(status)
        
        missing_items = res.get("smoothed_missing", [])
        if missing_items:
            lines.append(f"Missing: {', '.join(missing_items).title()}")
            
        color = (0, 255, 0) if "C - COMPLIANT" in status else (0, 0, 255)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        y_offset = max(10, y1 - (len(lines) * 20))
        for i, line in enumerate(lines):
            cv2.putText(image, line, (x1, y_offset + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    cv2.putText(image, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--zone", default="default")
    parser.add_argument("--output", default="outputs/demo_annotated.mp4")
    parser.add_argument("--frame-skip", type=int, default=None, help="Process every Nth frame")
    args = parser.parse_args()

    print("-" * 50)
    print(f"PyTorch version: {torch.__version__}")
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    device = "cuda:0" if has_cuda else "cpu"
    print("-" * 50)

    frame_skip = args.frame_skip
    if frame_skip is None:
        frame_skip = 1 if has_cuda else 2

    if not os.path.exists(args.video):
        return
        
    config = load_rules("config/rules.yaml")
    zone_rules = config["zones"].get(args.zone, config["zones"]["default"])
    thresholds = config.get("confidence_thresholds", {})
    
    ppe_path = "models/ppe.pt" if os.path.exists("models/ppe.pt") else "models/ppe_model.pt"
    if not os.path.exists(ppe_path): ppe_path = "yolo11n.pt"
    fire_path = "models/fire.pt" if os.path.exists("models/fire.pt") else "models/fire_model.pt"
    if not os.path.exists(fire_path): fire_path = "yolov8n.pt"
        
    try:
        detector = SafetyDetector(ppe_path, fire_path, device=device)
    except Exception as e:
        return
    
    smoother = TemporalSmoother(n_frames=15, m_frames=10, max_age=30)
    alert_mgr = AlertManager()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if input_fps == 0: input_fps = 30.0
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, input_fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    
    recent_fire_hazards = []
    last_results = []
    last_detections = []
    current_fps = 0.0
    
    # State tracker for dashboard
    demo_summary = {}
    
    print(f"Processing video: {args.video} ({width}x{height} @ {input_fps} FPS, {total_frames} frames)")
        
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame_count += 1
        recent_fire_hazards = [h for h in recent_fire_hazards if frame_count - h['frame'] <= 5]
        
        if frame_count % frame_skip != 0:
            out.write(draw_video_annotations(frame, last_results, last_detections, recent_fire_hazards, current_fps))
            continue
            
        processed_count += 1
        detections = detector.track(frame)
        
        fire_detections = detector.detect_fire(frame)
        for fd in fire_detections:
            fd['frame'] = frame_count
            recent_fire_hazards.append(fd)
            
        results = evaluate_compliance(detections, zone_rules, thresholds)
        
        for res in results:
            p = res["person"]
            required = res.get("required", [])
            
            if p.track_id is not None:
                smoothed_missing = smoother.update(p.track_id, res["missing"], frame_count)
                res["smoothed_missing"] = smoothed_missing
                res["smoothed_status"] = "NC - NON-COMPLIANT" if smoothed_missing else "C - COMPLIANT"
                
                if smoothed_missing:
                    details = f"Missing PPE: {', '.join(smoothed_missing)}"
                    alert_mgr.trigger_alert("demo_cam", "demo_zone", "PPE_VIOLATION", p.track_id, "HIGH", details, frame)
            else:
                res["smoothed_missing"] = res["missing"]
                res["smoothed_status"] = res["status"]
                
            res["smoothed_ppe"] = {}
            for item in required:
                res["smoothed_ppe"][item] = "MISSING" if item in res["smoothed_missing"] else "PRESENT"
                
            # Update summary dict for the dashboard
            if p.track_id is not None:
                demo_summary[str(p.track_id)] = {
                    "status": res["smoothed_status"],
                    "ppe": res["smoothed_ppe"],
                    "missing": res["smoothed_missing"]
                }
                
        smoother.cleanup(frame_count)
        
        elapsed = time.time() - start_time
        current_fps = processed_count / elapsed if elapsed > 0 else 0
        
        last_results = results
        last_detections = detections
        
        out.write(draw_video_annotations(frame, results, detections, recent_fire_hazards, current_fps))
        
        if frame_count == 1 or frame_count % 10 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
            
    total_time = time.time() - start_time
    cap.release()
    out.release()
    
    # Save the summary JSON for the dashboard
    with open("outputs/demo_summary.json", "w") as f:
        json.dump(demo_summary, f)
        
    print("="*50)
    print("FINAL REPORT")
    print("="*50)
    print(f"Output frames written: {frame_count}")
    print(f"Processing time: {total_time:.2f} seconds")
    print(f"Wrote demo_summary.json with {len(demo_summary)} tracked persons.")
    print("="*50)

if __name__ == "__main__":
    main()
