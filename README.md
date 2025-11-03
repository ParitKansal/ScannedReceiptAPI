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

You can either:

* Place your `.pt` model in the project root (default: `model.pt`)
* **Or use the provided script to download it automatically if the GitHub model doesn’t work:**

```bash
python download_model.py
```

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
  "filename": "random_crop_rot0_103_13_8.FRIENDS UNITED LTD - RECEIPTS (JUNE) 6_page_5_filled.jpg",
  "timestamp": "20251103_111250",
  "num_detections": 4,
  "detections": [
    {
      "confidence": 0.9708718061447144,
      "x1": 566.3211059570312,
      "y1": 0,
      "x2": 1160.4078369140625,
      "y2": 1151.1822509765625,
      "x_center": 0.390662656758166,
      "y_center": 0.2803658672617054,
      "w_norm": 0.26881752531992364,
      "h_norm": 0.5607317345234109
    },
    ...
  ]
}
```

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
├─ download_model.py       # Script to download model from Google Drive
├─ model.pt                # YOLOv9 model
└─ README.md
```