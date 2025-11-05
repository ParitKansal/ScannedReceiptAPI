# ScannedReceiptAPI: Receipt Detection API

A **FastAPI** server for detecting receipts using YOLO.
Supports uploading **one or multiple images** and returns **bounding boxes** in both **pixel coordinates** and **normalized format**.
Works on **Linux, macOS, and Windows**.


## 🚀 Requirements

* Python 3.10+
* Conda
* GPU optional (CUDA supported for acceleration)


## ⚙️ Installation

### **Option 1 — With Conda** (Recommended if you already use Conda)

1. **Clone the repository**

```bash
git clone https://github.com/ParitKansal/ScannedReceiptAPI
cd ScannedReceiptAPI
```

2. **Create Conda environment**

```bash
conda env create -f environment.yml
conda activate ScannedReceipt-fastapi
```

3. **Ensure model exists**

You can either:

* Place your YOLO `.pt` model in the project root (default: `model.pt`)
* **Or download automatically:**

```bash
python download_model.py
```

### **Option 2 — Without Conda** (Using `pip` + Virtual Environment)

1. **Clone the repository**

```bash
git clone https://github.com/ParitKansal/ScannedReceiptAPI
cd ScannedReceiptAPI
```

2. **Create & activate virtual environment**

```bash
python -m venv venv
```

* **Windows**

  ```bash
  venv\Scripts\activate
  ```

* **Mac/Linux**

  ```bash
  source venv/bin/activate
  ```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Ensure model exists**

```bash
python download_model.py
```

(Or manually place `model.pt` in project root.)

## ▶️ Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Swagger UI:**
[http://localhost:8000/docs](http://localhost:8000/docs)


## 🧠 API Usage

### Endpoint

`POST /predict`

* **Description:** Upload **one or more receipt images** and get detected bounding boxes.
* **Request:** `multipart/form-data` with field name `files`
* **Response (JSON):**

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


## 🐍 Example: Python Client

```python
import requests

url = "http://localhost:8000/predict"

# List of local image paths
file_paths = [
    "/path/to/receipt1.jpg",
    "/path/to/receipt2.jpg"
]

# Prepare multiple files for upload
files = [("files", open(path, "rb")) for path in file_paths]

# Send POST request
response = requests.post(url, files=files)

# Close file handles
for _, f in files:
    f.close()

# Print the JSON response
print(response.json())
```


## ✂️ Cropping Detected Receipts from Images

You can easily crop the detected bounding boxes from the original image using the prediction results.

```python
from PIL import Image
import requests
from io import BytesIO

# Example: single image crop based on API prediction
url = "http://localhost:8000/predict"
image_path = "/path/to/receipt.jpg"

# Send image
with open(image_path, "rb") as f:
    files = {"files": f}
    response = requests.post(url, files=files)

data = response.json()
detections = data["results"][0]["detections"]

# Open original image
img = Image.open(image_path)

# Loop through all detections and crop each region
for i, det in enumerate(detections):
    x1, y1, x2, y2 = map(int, [det["x1"], det["y1"], det["x2"], det["y2"]])
    cropped = img.crop((x1, y1, x2, y2))
    cropped.save(f"cropped_receipt_{i+1}.jpg")
    print(f"Saved cropped image: cropped_receipt_{i+1}.jpg")
```

### 🧩 Output

* Each detected region is saved as a new image file:

  ```
  cropped_receipt_1.jpg
  cropped_receipt_2.jpg
  ...
  ```
* You can further process or OCR these cropped regions for text extraction.


## 📂 Folder Structure

```
ScannedReceiptAPI/
│
├─ app/
│   ├─ main.py             # FastAPI app entry point
│   ├─ predict.py          # YOLO prediction & post-processing logic
│   └─ core/
│       └─ config.py       # Configuration (model path, thresholds, etc.)
│
├─ environment.yml         # Conda environment file
├─ download_model.py       # Optional model download script
├─ model.pt                # YOLO model weights
└─ README.md
```
