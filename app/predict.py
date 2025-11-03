from fastapi import UploadFile
from ultralytics import YOLO
from app.core.config import get_settings
from datetime import datetime
from io import BytesIO
from PIL import Image

settings = get_settings()
model = YOLO(settings.MODEL_PATH)

def predict_image(file: UploadFile):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load image in memory using PIL
    img_bytes = file.file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    width, height = img.size

    # Run prediction directly on PIL image
    results = model.predict(
        source=img,
        conf=settings.CONF_THRESH,
        imgsz=settings.IMG_SIZE,
        max_det=settings.MAX_DET,
        save=False  # do NOT save images
    )

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # normalized coordinates
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w_norm = (x2 - x1) / width
            h_norm = (y2 - y1) / height

            detections.append({
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_normalized": [x_center, y_center, w_norm, h_norm]
            })

    return {
        "filename": file.filename,
        "timestamp": timestamp,
        "num_detections": len(detections),
        "detections": detections
    }
