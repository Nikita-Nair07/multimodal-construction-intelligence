
import json
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from io import BytesIO
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

load_dotenv()

router = APIRouter(prefix="/ai", tags=["AI Integration"])

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


# Activity 4.2: Structured output contracts

class AIAnalysis(BaseModel):
    task: Literal["safety", "crack", "planning", "valuation"]
    observations: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    risk_level: Literal[
        "low", "moderate", "high", "critical", "uncertain"
    ] = "uncertain"
    limitations: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    success: bool
    analysis: AIAnalysis
    disclaimer: str


# Activity 4.1 and 4.3: Multimodal and context-aware reasoning

@router.post("/analyze", response_model=AnalysisResponse)
async def multimodal_analysis(
    task: Literal["safety", "crack", "planning", "valuation"] = Form(...),
    description: str = Form(""),
    file: UploadFile = File(...),
):
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured.",
        )

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400,
            detail="Upload a JPG, PNG, or WEBP image.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    task_instructions = {
        "safety": (
            "Identify visible construction-site safety hazards and PPE concerns."
        ),
        "crack": (
            "Describe visible crack patterns and possible risks. Do not claim "
            "structural safety or diagnose the cause from an image alone."
        ),
        "planning": (
            "Describe visible site conditions relevant to preliminary project "
            "planning. Do not invent costs, approvals, or exact durations."
        ),
        "valuation": (
            "Describe visible property features that may influence value. "
            "Do not estimate market value without location and market data."
        ),
    }

    prompt = f"""
    You are assisting with preliminary construction intelligence.

    Selected task: {task}
    Task instructions: {task_instructions[task]}
    User's contextual description: {description or "No description supplied."}

    Consider the image and description together. Distinguish visible facts
    from assumptions. Mention uncertainty where evidence is insufficient.
    Do not invent measurements, prices, hidden conditions, or certifications.

    Return valid JSON with exactly these fields:
    {{
      "task": "{task}",
      "observations": ["visible observations"],
      "findings": ["task-specific findings"],
      "recommendations": ["practical next steps"],
      "risk_level": "low, moderate, high, critical, or uncertain",
      "limitations": ["important uncertainty or missing information"]
    }}
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
                response_schema=AIAnalysis,
                temperature=0.2,
            ),
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="AI returned an empty response.",
            )

        analysis = AIAnalysis.model_validate_json(response.text)

        return AnalysisResponse(
            success=True,
            analysis=analysis,
            disclaimer=(
                "AI-generated preliminary analysis only. Professional "
                "inspection and verification may be required."
            ),
        )

    except HTTPException:
        raise
    except (ValidationError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"AI response failed schema validation: {error}",
        )
    except Exception as error:
        print(f"Multimodal analysis error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=502,
            detail="AI analysis is temporarily unavailable or failed.",
        )
    finally:
        await file.close()