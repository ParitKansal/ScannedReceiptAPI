from fastapi import UploadFile
from ultralytics import YOLO
from app.core.config import get_settings
from datetime import datetime
from io import BytesIO
from PIL import Image
import pandas as pd

settings = get_settings()
model = YOLO(settings.MODEL_PATH)

# --- Helper functions for merging boxes ---
def intersection_area(b1, b2):
    x1 = max(b1["x1"], b2["x1"])
    y1 = max(b1["y1"], b2["y1"])
    x2 = min(b1["x2"], b2["x2"])
    y2 = min(b1["y2"], b2["y2"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_area(b):
    return max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])

def merge_boxes(b_keep, b_other):
    return {
        "confidence": max(b_keep["confidence"], b_other["confidence"]),
        "x1": min(b_keep["x1"], b_other["x1"]),
        "y1": min(b_keep["y1"], b_other["y1"]),
        "x2": max(b_keep["x2"], b_other["x2"]),
        "y2": max(b_keep["y2"], b_other["y2"]),
    }

def merge_boxes_iterative(df, containment_threshold=0.9, max_iter=10):
    df = df.copy().reset_index(drop=True)
    for _ in range(max_iter):
        merged_any = False
        merged_rows = []
        used = [False] * len(df)
        for i in range(len(df)):
            if used[i]:
                continue
            row_i = df.loc[i].to_dict()
            for j in range(i + 1, len(df)):
                if used[j]:
                    continue
                row_j = df.loc[j].to_dict()
                inter = intersection_area(row_i, row_j)
                min_area = max(min(box_area(row_i), box_area(row_j)), 1e-9)
                if inter / min_area >= containment_threshold:
                    row_i = merge_boxes(row_i, row_j)
                    used[j] = True
                    merged_any = True
            merged_rows.append(row_i)
            used[i] = True
        new_df = pd.DataFrame(merged_rows)
        if not merged_any:
            break
        df = new_df
    return df

# --- Prediction function ---
def predict_image(file: UploadFile, save_input=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load image in memory
    img_bytes = file.file.read()
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    width, height = img.size

    # Save input copy if requested
    if save_input:
        from pathlib import Path
        out_dir = Path("pred_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        input_path = out_dir / f"input_{timestamp}_{file.filename}"
        with open(input_path, "wb") as f:
            f.write(img_bytes)

    # Run YOLO prediction
    results = model.predict(
        source=img,
        conf=settings.CONF_THRESH,
        imgsz=settings.IMG_SIZE,
        max_det=settings.MAX_DET,
        save=False
    )

    # Extract bounding boxes
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x_center": (x1 + x2) / 2 / width,
                "y_center": (y1 + y2) / 2 / height,
                "w_norm": (x2 - x1) / width,
                "h_norm": (y2 - y1) / height
            })

    # Convert to DataFrame and merge boxes iteratively
    df = pd.DataFrame(detections)
    if not df.empty:
        df_merged = merge_boxes_iterative(df, containment_threshold=0.9)
        detections = df_merged.to_dict(orient="records")

    return {
        "filename": file.filename,
        "timestamp": timestamp,
        "num_detections": len(detections),
        "detections": detections
    }
