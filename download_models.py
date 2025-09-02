# download_models.py
from ultralytics import YOLO
import os

def download_document_models():
    """Download specialized document layout models"""
    
    print("📥 Downloading document layout models...")
    
    # Document layout model from Hugging Face
    try:
        model = YOLO("hf://DILHTWD/yolov8-doclaynet")  
        print("✅ Downloaded DocLayNet YOLOv8 model")
        return model
    except:
        print("⚠️ DocLayNet model unavailable, trying alternatives...")
    
    # Alternative: YOLOv8 segmentation model
    try:
        model = YOLO("yolov8n-seg.pt")  # Segmentation version
        print("✅ Downloaded YOLOv8 segmentation model")
        return model
    except:
        print("❌ Model download failed")
        return None

# Download and test
model = download_document_models()
if model:
    print(f"📋 Model classes: {list(model.names.values())}")
