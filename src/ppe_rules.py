import yaml
from typing import List, Dict, Any, Tuple
from .detectors import Detection

def load_rules(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def evaluate_compliance(detections: List[Detection], zone_rules: dict, thresholds: dict) -> List[Dict[str, Any]]:
    persons = [d for d in detections if d.cls == "person" and d.conf >= thresholds.get("person", 0.5)]
    gear = [d for d in detections if d.cls in ["hardhat", "vest", "gloves", "boots"]]
    
    required_ppe = zone_rules.get("required_ppe", [])
    results = []
    
    for person in persons:
        px1, py1, px2, py2 = person.bbox
        w = px2 - px1
        h = py2 - py1
        
        regions = {
            "hardhat": (px1 - 0.10*w, py1 - 0.10*h, px2 + 0.10*w, py1 + 0.28*h),
            "vest": (px1 - 0.10*w, py1 + 0.22*h, px2 + 0.10*w, py1 + 0.68*h),
            "gloves": (px1 - 0.20*w, py1 + 0.35*h, px2 + 0.20*w, py1 + 0.75*h),
            "boots": (px1 - 0.10*w, py1 + 0.78*h, px2 + 0.10*w, py2 + 0.10*h)
        }
        
        associated_gear = []
        for g in gear:
            if g.cls in regions:
                rx1, ry1, rx2, ry2 = regions[g.cls]
                cx, cy = get_centroid(g.bbox)
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    associated_gear.append(g.cls)
        
        missing = [item for item in required_ppe if item not in associated_gear]
        status = "VIOLATION" if missing else "COMPLIANT"
        
        results.append({
            "person": person,
            "status": status,
            "missing": missing,
            "associated": associated_gear
        })
        
    return results
