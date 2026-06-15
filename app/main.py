from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

from app.model import predict_image

app = FastAPI(
    title="Crop Disease Detection API",
    description="Serve the EfficientNet-B3 crop disease classifier via FastAPI",
    version="1.0.0",
)


@app.get("/")
def root():
    return {"message": "Crop Disease Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # handle (prediction, confidence) or (prediction, confidence, index)
    prediction, confidence, *_ = predict_image(image)

    return {
        "filename": file.filename,
        "prediction": prediction,
        "confidence": round(confidence, 4),
    }
