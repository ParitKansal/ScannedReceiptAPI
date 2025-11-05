# ScannedReceiptAPI: Receipt Detection API

This repository provides a FastAPI-based server for detecting receipts in images using a YOLO object detection model.
The API supports uploading one or multiple images and returns bounding boxes in both pixel coordinates and normalized format.
Works on Linux, macOS, and Windows. GPU acceleration is supported where available.



## Requirements

* Python 3.10+
* (Optional) Conda
* (Optional) NVIDIA GPU with CUDA support for faster inference



## Installation

### Option 1 — Using Conda (If Conda is available)

1. Clone the repository:

```bash
git clone https://github.com/ParitKansal/ScannedReceiptAPI
cd ScannedReceiptAPI
```

2. Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate ScannedReceipt-fastapi
```

3. Ensure that the YOLO model file is available:

   * Place your `.pt` model in the project root (default filename: `model.pt`), or
   * Download it automatically:

```bash
python download_model.py
```



### Option 2 — Without Conda (Using pip and Virtual Environment)

1. Clone the repository:

```bash
git clone https://github.com/ParitKansal/ScannedReceiptAPI
cd ScannedReceiptAPI
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
```

* Windows:

```bash
venv\Scripts\activate
```

* macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Ensure the YOLO model file exists:

```bash
python download_model.py
```

Alternatively, manually place `model.pt` in the project root.



## Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation will be available at:

```
http://localhost:8000/docs
```



## API Usage

### Endpoint

`POST /predict`

**Description:**
Uploads one or more images and performs receipt detection.

**Request Format:**
`multipart/form-data` with field name `files`

**Response Example:**

```json
{
  "results": [
    {
      "filename": "receipt1.jpg",
      "timestamp": "20251103_132010",
      "num_detections": 3,
      "detections": [
        {
          "confidence": 0.97,
          "x1": 566.32,
          "y1": 0.0,
          "x2": 1160.40,
          "y2": 1151.18,
          "x_center": 0.39,
          "y_center": 0.28,
          "w_norm": 0.26,
          "h_norm": 0.56
        }
      ]
    }
  ]
}
```



## Example: Python Client Code

```python
import requests

url = "http://localhost:8000/predict"

file_paths = [
    "/path/to/receipt1.jpg",
    "/path/to/receipt2.jpg"
]

files = [("files", open(path, "rb")) for path in file_paths]

response = requests.post(url, files=files)

for _, f in files:
    f.close()

print(response.json())
```



## Cropping Detected Receipts

```python
from PIL import Image
import requests
from io import BytesIO

url = "http://localhost:8000/predict"
image_path = "/path/to/receipt.jpg"

with open(image_path, "rb") as f:
    files = {"files": f}
    response = requests.post(url, files=files)

data = response.json()
detections = data["results"][0]["detections"]

img = Image.open(image_path)

for i, det in enumerate(detections):
    x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(f"cropped_receipt_{i+1}.jpg")
    print(f"Saved cropped_receipt_{i+1}.jpg")
```



## Project Structure

```
ScannedReceiptAPI/
│
├─ app/
│   ├─ main.py               # FastAPI application entry point
│   ├─ predict.py            # YOLO inference and post-processing logic
│   └─ core/
│       └─ config.py         # Configuration (model paths, thresholds, etc.)
│
├─ environment.yml           # Conda environment file
├─ requirements.txt          # pip requirements (if needed)
├─ download_model.py         # Optional model download script
├─ model.pt                  # YOLO model weights file (optional, user-provided)
└─ README.md
```


