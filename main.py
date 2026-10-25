
import os
import json
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from google import genai
from google.genai import types

# Load environment variables from backend/.env
load_dotenv()

# Initialize Gemini client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# Create FastAPI application
app = FastAPI(
    title="Multimodal Construction Intelligence Platform",
    description=(
        "AI-powered construction safety image analysis API. "
        "Upload a construction-site image to identify visible hazards."
    ),
    version="1.0.0",
)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Multimodal Construction Intelligence Platform API is running",
        "docs": "/docs",
        "health": "/health",
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": client is not None,
    }


# Construction safety image analysis endpoint
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is missing. Check your backend/.env file.",
        )

    # Check uploaded file type
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format. Upload a JPG, PNG, or WEBP image.",
        )

    # Read image bytes
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # Limit uploads to 10 MB
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    # Validate that the file is a readable image
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    # Prompt for construction-site safety analysis
    prompt = """
    Analyze this construction-site image for visible safety hazards.

    Return a JSON object with these fields:
    {
      "site_observations": ["visible observations"],
      "hazards": [
        {
          "hazard": "description of the visible hazard",
          "severity": "Low, Medium, High, or Critical",
          "reason": "why this may be dangerous",
          "recommended_action": "practical corrective action"
        }
      ],
      "ppe_compliance": [
        "observations about visible personal protective equipment"
      ],
      "housekeeping": "observations about site cleanliness and organization",
      "overall_risk": "Low, Medium, High, or Critical",
      "summary": "brief summary of the image"
    }

    Important rules:
    - Report only what is visibly supported by the image.
    - Do not claim that missing equipment or hazards exist if they cannot
      be confirmed visually.
    - If something is unclear, explicitly state that it cannot be determined.
    - If no obvious hazards are visible, say so.
    - This is an AI-assisted preliminary review, not a certified safety
      inspection.
    - Return valid JSON only, without Markdown code fences.
    """

    # Call Gemini to analyze the image
    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=file.content_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        result_text = response.text

        if not result_text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response. Try another image.",
            )

        # Convert the model response into JSON
        try:
            analysis = json.loads(result_text)
        except json.JSONDecodeError:
            analysis = {"raw_analysis": result_text}

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "analysis": analysis,
                "disclaimer": (
                    "AI-generated preliminary observations. "
                    "A qualified safety professional must verify findings."
                ),
            }
        )

    except HTTPException:
        raise
    except Exception as error:
        print(f"Gemini analysis error: {error}")
        raise HTTPException(
            status_code=502,
            detail=f"{type(error).__name__}: {error}",
        )

    finally:
        await file.close()
