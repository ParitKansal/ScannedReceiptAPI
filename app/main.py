from fastapi import FastAPI, UploadFile, File
from typing import List
from app.predict import predict_image

app = FastAPI(title="YOLO-FastAPI Receipt Detection")

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    """
    Upload one or multiple receipt images and get bounding boxes in JSON
    """
    results = []
    for file in files:
        result = predict_image(file)
        results.append(result)
    return {"results": results}
