import math
from typing import List, Dict, Any

class FireTemporalSmoother:
    def __init__(self, persistence_frames=5, escalation_frames=10):
        self.history = []
        self.persistence_frames = persistence_frames
        self.escalation_frames = escalation_frames
        
    def update(self, new_hazards: List[Dict[str, Any]], frame_count: int) -> List[Dict[str, Any]]:
        # Add new hazards to history
        for h in new_hazards:
            h["frame"] = frame_count
            self.history.append(h)
            
        # Remove hazards older than persistence_frames
        self.history = [h for h in self.history if frame_count - h["frame"] <= self.persistence_frames]
        
        # We can group overlapping bounding boxes if necessary, but for simplicity:
        # If there are fire hazards consistently for escalation_frames, we bump severity.
        return self.history

def evaluate_fire_hazards(fire_detections: List[Dict[str, Any]], persons: List[Any], frame_count: int, smoother: FireTemporalSmoother) -> List[Dict[str, Any]]:
    hazards = fire_detections
    if not hazards:
        # Still update smoother to clear old ones
        return smoother.update([], frame_count)
        
    events = []
    for hazard in hazards:
        hx1, hy1, hx2, hy2 = hazard["bbox"]
        area = (hx2 - hx1) * (hy2 - hy1)
        if area < 400:
            continue
            
        h_width = max(hx2 - hx1, 50)
        h_center = ((hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0)
        
        at_risk_count = 0
        at_risk_ids = []
        
        for p in persons:
            px1, py1, px2, py2 = p.bbox
            feet_pt = ((px1 + px2) / 2.0, py2)
            
            dist = math.hypot(feet_pt[0] - h_center[0], feet_pt[1] - h_center[1])
            norm_dist = dist / h_width
            
            if norm_dist <= 2.0:
                at_risk_count += 1
                if p.track_id is not None:
                    at_risk_ids.append(p.track_id)
                    
        cls_name = hazard.get("cls", "fire")
        if cls_name == "smoke":
            severity = "WARNING"
            details = "Smoke detected."
        else:
            if at_risk_count > 0:
                severity = "CRITICAL"
                details = f"Fire with {at_risk_count} workers nearby! IDs: {at_risk_ids}"
            else:
                severity = "HIGH"
                details = "Isolated fire detected."
                
        events.append({
            "hazard": hazard,
            "severity": severity,
            "at_risk_count": at_risk_count,
            "details": details,
            "cls": cls_name,
            "bbox": hazard["bbox"]
        })
        
    smoothed_events = smoother.update(events, frame_count)
    return smoothed_events
