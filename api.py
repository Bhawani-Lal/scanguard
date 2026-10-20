import io
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from predict import load_model, load_threshold, predict, looks_like_casting

app = FastAPI(
    title="ScanGuard API",
    description="Classifies casting images as DEFECTIVE or OK.",
    version="1.0",
)

# Load once at startup, then reuse for every request
model = load_model()
threshold = load_threshold()

REVIEW_BAND = (0.20, 0.60)


@app.get("/health")
def health():
    """Simple check that the service is running."""
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Upload an image; get back the prediction as JSON."""
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="File is not a valid image.")

    if not looks_like_casting(image):
        raise HTTPException(status_code=422, detail="Image does not look like a casting inspection photo.")

    label, p_def = predict(image, model, threshold)

    return {
        "filename": file.filename,
        "prediction": label,
        "defect_probability": round(p_def, 4),
        "threshold": threshold,
        "needs_manual_review": REVIEW_BAND[0] <= p_def <= REVIEW_BAND[1],
    }