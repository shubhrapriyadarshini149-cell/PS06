import math
from typing import List, Dict, Any
from .detectors import Detection

def evaluate_fire_hazards(detections: List[Detection], persons: List[Detection]) -> List[Dict[str, Any]]:
    hazards = [d for d in detections if d.cls in ["fire", "smoke"]]
    if not hazards:
        return []
        
    events = []
    
    for hazard in hazards:
        # Check size (ignore tiny sparks)
        hx1, hy1, hx2, hy2 = hazard.bbox
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
            
            # Distance from feet to fire center
            dist = math.hypot(feet_pt[0] - h_center[0], feet_pt[1] - h_center[1])
            norm_dist = dist / h_width
            
            if norm_dist <= 2.0:
                at_risk_count += 1
                if p.track_id is not None:
                    at_risk_ids.append(p.track_id)
                    
        if hazard.cls == "smoke":
            severity = "WARNING"
            details = "Smoke detected."
        else: # fire
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
            "details": details
        })
        
    return events
