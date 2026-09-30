from ultralytics import YOLO
import sys
import os

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    print(f"--- Inspecting {model_path} ---")
    model = YOLO(model_path)
    print("Classes:", model.names)
    print("---------------------------------")

if __name__ == "__main__":
    inspect_model("models/ppe_model.pt")
    inspect_model("models/fire_model.pt")
