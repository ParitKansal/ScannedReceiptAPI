from fastapi import FastAPI, UploadFile
from app.predict import predict_image

app = FastAPI(title="YOLO-FastAPI Receipt Detection")

@app.post("/predict")
async def predict(file: UploadFile):
    """
    Upload a receipt image and get bounding boxes in JSON
    """
    return predict_image(file)
