# ScannedReceiptAPI: Receipt Detection API

A **FastAPI** server for detecting receipts.
Supports uploading images and returning **bounding boxes** in both **pixel coordinates** and **normalized format**. Works on **Linux, macOS, and Windows**.

## Requirements

* Python 3.10+
* Conda
* GPU optional 

## Installation

1. **Clone the repository**

```bash
gh repo clone ParitKansal/ScannedReceiptAPI
cd ScannedReceiptAPI
```

2. **Create Conda environment**

```bash
conda env create -f environment.yml
conda activate ScannedReceipt-fastapi
```

3. **Ensure model exists**

Place your `.pt` model in the project root (default: `model.pt`).
You can also update `app/core/config.py` to use a custom model path.

## Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)


## API Usage

### Endpoint

`POST /predict`

* **Description:** Upload an image and get detected receipts
* **Request:** `multipart/form-data` with `file` field (image)
* **Response (JSON):**

```json
{
  "filename": "receipt.jpg",
  "timestamp": "20251103_101234",
  "num_detections": 4,
  "detections": [
    {
      "class_id": 0,
      "class_name": "receipt",
      "confidence": 0.87,
      "bbox_xyxy": [102.4, 215.1, 389.7, 610.9],
      "bbox_normalized": [0.39, 0.65, 0.47, 0.62]
    }
  ]
}
```

* `bbox_xyxy`: pixel coordinates `[x1, y1, x2, y2]`
* `bbox_normalized`: `[x_center, y_center, width, height]` normalized 0–1

### Example: Python client

```python
import requests

url = "http://localhost:8000/predict"
with open("receipt.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())
```

## Folder Structure

```
ScannedReceiptAPI/
│
├─ app/
│   ├─ main.py             # FastAPI app
│   ├─ predict.py          # YOLO prediction logic
│   └─ core/
│       └─ config.py       # Model & output configuration
│
├─ environment.yml         # Conda environment file
├─ best-2.pt               # YOLOv9 model
└─ README.md
```
