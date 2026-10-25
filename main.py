
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


from pydantic import BaseModel, Field
from typing import Literal


class CrackFinding(BaseModel):
    crack_type: Literal[
        "hairline", "vertical", "diagonal", "structural", "other", "uncertain"
    ]
    severity: Literal["minor", "moderate", "severe", "uncertain"]
    probable_causes: list[str] = Field(default_factory=list)
    potential_risks: list[str] = Field(default_factory=list)
    recommended_repairs: list[str] = Field(default_factory=list)
    preventive_measures: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    confidence_note: str


@app.post("/analyze-crack")
async def analyze_crack(
    file: UploadFile = File(...),
    description: str = "",
):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key is not configured.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image size must not exceed 10 MB.",
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    prompt = f"""
    You are assisting with preliminary visual inspection of cracks in buildings.

    User description: {description or "No description provided."}

    Examine the attached image and return JSON matching this schema:
    {{
      "crack_type": "hairline, vertical, diagonal, structural, other, or uncertain",
      "severity": "minor, moderate, severe, or uncertain",
      "probable_causes": ["possible causes, clearly marked as hypotheses"],
      "potential_risks": ["potential concerns"],
      "recommended_repairs": ["appropriate next steps or repair considerations"],
      "preventive_measures": ["relevant preventive measures"],
      "observations": ["visible features only"],
      "confidence_note": "limitations and uncertainty"
    }}

    Rules:
    - Do not infer structural safety from a photograph alone.
    - Do not present probable causes as confirmed facts.
    - Do not invent crack width, depth, or measurements.
    - If scale or image quality is inadequate, state this.
    - If the crack cannot be assessed, use uncertain classifications.
    - Recommend prompt professional inspection for severe-looking cracks,
      widening cracks, displacement, or signs of structural distress.
    - Return valid JSON only.
    """

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
                response_schema=CrackFinding,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response. Try again.",
            )

        finding = CrackFinding.model_validate_json(response.text)

        return {
            "success": True,
            "filename": file.filename,
            "crack_analysis": finding.model_dump(),
            "disclaimer": (
                "Preliminary AI-assisted visual assessment only. "
                "A qualified structural engineer must evaluate the structure."
            ),
        }

    except HTTPException:
        raise
    except Exception as error:
        print(f"Crack analysis error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=502,
            detail=f"Crack analysis failed: {type(error).__name__}: {error}",
        )
    finally:
        await file.close()


from typing import Literal
from pydantic import BaseModel, Field


# ==========================================
# ACTIVITY 3.1: CONSTRUCTION PROJECT PLANNING
# ==========================================

class ProjectPlanningRequest(BaseModel):
    project_type: Literal[
        "residential", "commercial", "renovation"
    ]
    area_sqft: float = Field(gt=0, le=10000000)
    budget_inr: float = Field(gt=0)
    workers: int = Field(ge=1, le=10000)
    start_date: str = Field(
        description="Planned start date in YYYY-MM-DD format"
    )


@app.post("/plan-project")
def plan_project(request: ProjectPlanningRequest):
    # Illustrative planning estimates, not an engineered schedule.
    phases = [
        {"phase": "Planning and approvals", "days": 14},
        {"phase": "Site preparation", "days": 7},
        {"phase": "Foundation", "days": 21},
        {"phase": "Structural work", "days": 45},
        {"phase": "Electrical and plumbing", "days": 21},
        {"phase": "Finishing", "days": 30},
        {"phase": "Inspection and handover", "days": 7},
    ]

    # Basic scale adjustment for project size and workforce.
    size_factor = max(0.75, min(2.5, request.area_sqft / 1500))
    workforce_factor = max(0.6, min(1.5, 8 / request.workers))

    planned_days = sum(p["days"] for p in phases)
    estimated_days = round(
        planned_days * size_factor * workforce_factor
    )

    budget_per_sqft = request.budget_inr / request.area_sqft

    return {
        "success": True,
        "activity": "3.1 Construction Project Planning Logic",
        "project": request.model_dump(),
        "estimated_duration_days": estimated_days,
        "phases": phases,
        "budget_per_sqft_inr": round(budget_per_sqft, 2),
        "planning_notes": [
            "Confirm local permits and site conditions before work begins.",
            "Schedule depends on design, materials, weather, and contractor capacity.",
            "Phase durations are illustrative and should be reviewed by a professional.",
        ],
        "risk_factors": [
            "Material delivery delays",
            "Weather conditions",
            "Permit and inspection delays",
            "Budget overruns and labour availability",
        ],
        "disclaimer": (
            "Preliminary planning estimate only, not a contractor-approved schedule."
        ),
    }


# ==========================================
# ACTIVITY 3.2: REAL ESTATE VALUATION ENGINE
# ==========================================

class PropertyValuationRequest(BaseModel):
    location: str = Field(min_length=2, max_length=200)
    property_type: Literal[
        "apartment", "independent_house", "commercial", "plot"
    ]
    area_sqft: float = Field(gt=0, le=10000000)
    price_per_sqft_inr: float = Field(gt=0)
    property_age_years: int = Field(ge=0, le=300)
    bedrooms: int = Field(ge=0, le=100)
    bathrooms: int = Field(ge=0, le=100)


@app.post("/valuate-property")
def valuate_property(request: PropertyValuationRequest):
    # Use the user's supplied local price benchmark.
    base_value = request.area_sqft * request.price_per_sqft_inr

    # Simple illustrative age adjustment; not a market appraisal.
    age_adjustment = min(request.property_age_years * 0.01, 0.30)
    estimated_value = base_value * (1 - age_adjustment)

    # A simple 10% range around the adjusted estimate.
    lower_value = estimated_value * 0.90
    upper_value = estimated_value * 1.10

    return {
        "success": True,
        "activity": "3.2 Real Estate Valuation Engine",
        "property": request.model_dump(),
        "base_value_inr": round(base_value, 2),
        "age_adjustment_percent": round(age_adjustment * 100, 2),
        "estimated_value_inr": round(estimated_value, 2),
        "indicative_range_inr": {
            "low": round(lower_value, 2),
            "high": round(upper_value, 2),
        },
        "valuation_method": (
            "Area multiplied by user-provided price per square foot, "
            "with a simple illustrative age adjustment."
        ),
        "limitations": [
            "The supplied price per square foot is not independently verified.",
            "Location, title, condition, floor, amenities, and comparable sales "
            "can materially affect market value.",
            "The range is illustrative, not a statistically derived confidence interval.",
        ],
        "disclaimer": (
            "Indicative estimate only, not a certified valuation or current market quote."
        ),
    }

from ai_integration import router as ai_router

app.include_router(ai_router)