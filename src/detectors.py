from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
from ultralytics import YOLO

# Canonical Class Normalizer
CLASS_MAP = {
    "helmet": "hardhat", "hard hat": "hardhat", "hardhat": "hardhat", 
    "white-helmet": "hardhat", "yellow-helmet": "hardhat",
    "vest": "vest", "safety vest": "vest", "safety-vest": "vest", "reflective-jacket": "vest",
    "gloves": "gloves", "safety gloves": "gloves",
    "boots": "boots", "safety boots": "boots", "shoes": "boots",
    "person": "person", "worker": "person",
    "fire": "fire", "flame": "fire",
    "smoke": "smoke"
}

@dataclass
class Detection:
    cls: str
    raw_cls: str
    conf: float
    bbox: Tuple[int, int, int, int]
    track_id: int = None

class SafetyDetector:
    def __init__(self, ppe_model_path: str, fire_model_path: str, device: str = None):
        self.ppe_model = YOLO(ppe_model_path)
        self.fire_model = YOLO(fire_model_path)
        self.device = device
        
        if self.device:
            self.ppe_model.to(self.device)
            self.fire_model.to(self.device)
        
    def _process_results(self, results, conf_threshold=0.3) -> List[Detection]:
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                raw_cls = result.names[int(box.cls[0])].lower()
                norm_cls = CLASS_MAP.get(raw_cls, raw_cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else None
                detections.append(Detection(
                    cls=norm_cls, raw_cls=raw_cls, conf=conf, bbox=(x1, y1, x2, y2), track_id=track_id
                ))
        return detections

    def predict(self, image, conf_threshold=0.3) -> List[Detection]:
        ppe_results = self.ppe_model(image, verbose=False, device=self.device)
        fire_results = self.fire_model(image, verbose=False, device=self.device)
        
        all_detections = []
        all_detections.extend(self._process_results(ppe_results, conf_threshold))
        all_detections.extend(self._process_results(fire_results, conf_threshold))
        
        return all_detections
        
    def track(self, image, conf_threshold=0.3) -> List[Detection]:
        # Track PPE model to assign track_ids to persons
        ppe_results = self.ppe_model.track(image, persist=True, tracker="bytetrack.yaml", conf=conf_threshold, verbose=False, device=self.device)
        return self._process_results(ppe_results, conf_threshold)

    def detect_fire(self, image, conf_threshold=0.3) -> List[Dict[str, Any]]:
        fire_results = self.fire_model(image, verbose=False, device=self.device)
        detections = self._process_results(fire_results, conf_threshold)
        
        return [
            {
                "cls": d.cls,
                "conf": d.conf,
                "bbox": list(d.bbox)
            }
            for d in detections if d.cls in ["fire", "smoke"]
        ]
