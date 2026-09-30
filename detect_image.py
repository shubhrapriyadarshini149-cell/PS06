import cv2
import argparse
import os
from src.detectors import SafetyDetector
from src.ppe_rules import load_rules, evaluate_compliance

def draw_annotations(image, results, detections, hazards):
    # Draw Hazards
    for hazard in hazards:
        x1, y1, x2, y2 = hazard.bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(image, f"{hazard.cls.upper()} ({hazard.conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
    # Draw Persons and PPE Status
    for res in results:
        p = res["person"]
        x1, y1, x2, y2 = p.bbox
        status = res["status"]
        
        if status == "COMPLIANT":
            color = (0, 255, 0)
            label = "COMPLIANT"
        else:
            color = (0, 0, 255)
            label = f"VIOLATION: Missing {', '.join(res['missing'])}"
            
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--zone", default="default")
    parser.add_argument("--output", default="outputs/snapshots/result.jpg")
    args = parser.parse_args()

    # Load Config
    config = load_rules("config/rules.yaml")
    zone_rules = config["zones"].get(args.zone, config["zones"]["default"])
    thresholds = config.get("confidence_thresholds", {})
    
    # Initialize Detector
    # Using generic yolov8n as placeholder if models/ don't exist yet
    ppe_path = "models/ppe_model.pt" if os.path.exists("models/ppe_model.pt") else "yolo11n.pt"
    fire_path = "models/fire_model.pt" if os.path.exists("models/fire_model.pt") else "yolov8n.pt"
    
    detector = SafetyDetector(ppe_path, fire_path)
    
    # Read Image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Failed to load image {args.image}")
        exit(1)
        
    # Detect
    detections = detector.predict(img)
    
    # Evaluate
    results = evaluate_compliance(detections, zone_rules, thresholds)
    hazards = [d for d in detections if d.cls in ["fire", "smoke"]]
    
    # Render
    out_img = draw_annotations(img, results, detections, hazards)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, out_img)
    print(f"Saved to {args.output}")
