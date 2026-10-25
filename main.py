
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from io import BytesIO

app = FastAPI(
    title="Multimodal Construction Intelligence Platform"
)

@app.get("/")
def home():
    return {
        "message": "Construction Intelligence API is running"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image."
        )

    contents = await file.read()

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be smaller than 10 MB."
        )

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    return {
        "filename": file.filename,
        "message": "Image uploaded and validated successfully.",
        "next_step": "Connect an AI vision model for construction analysis."
    }
