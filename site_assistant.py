
from datetime import date
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/assistant", tags=["Site Assistant"])


# Activity 5.1: Construction Chatbot

class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


@router.post("/chat")
def construction_chatbot(request: ChatRequest):
    question = request.question.lower()

    answers = [
        (["crack", "cracks", "structural damage"],
         "Keep people away from potentially unsafe areas. "
         "Arrange an inspection by a qualified structural engineer. "
         "This prototype cannot determine structural safety."),
        (["helmet", "ppe", "safety equipment"],
         "Workers should use appropriate helmets, safety footwear, "
         "high-visibility clothing and task-specific PPE."),
        (["concrete", "curing"],
         "Concrete curing depends on the mix, specifications, "
         "weather and site conditions. Follow the approved method "
         "statement and advice from the responsible engineer."),
        (["cement", "storage"],
         "Store cement in a dry, covered place, protected from "
         "moisture, and follow the supplier's storage guidance."),
        (["rain", "rainfall"],
         "Protect materials and excavations from water. Review "
         "drainage, access routes and the approved site safety plan."),
        (["excavation", "trench"],
         "Excavations require a competent-person assessment, "
         "appropriate protective systems and checks for underground "
         "utilities. Never enter an unsafe excavation."),
        (["budget", "cost", "estimate"],
         "Construction costs depend on location, design, materials, "
         "labour and current rates. Obtain a project-specific estimate "
         "from a qualified professional."),
    ]

    for keywords, answer in answers:
        if any(word in question for word in keywords):
            return {
                "activity": "5.1",
                "question": request.question,
                "answer": answer,
                "prototype": True,
            }

    return {
        "activity": "5.1",
        "question": request.question,
        "answer": (
            "Please specify the construction topic, such as site "
            "safety, materials, concrete, excavation or project costs. "
            "For safety-critical decisions, consult a qualified professional."
        ),
        "prototype": True,
    }


# Activity 5.2: Daily Site Assistant

class DailySiteRequest(BaseModel):
    project_type: str = Field(min_length=2, max_length=100)
    workers: int = Field(ge=0, le=10000)
    work_planned: str = Field(min_length=2, max_length=1000)
    weather: Literal["sunny", "cloudy", "rainy", "hot", "windy"]
    issues: str = Field(default="None reported", max_length=1000)


@router.post("/daily-site")
def daily_site_assistant(request: DailySiteRequest):
    checklist = [
        "Conduct the site safety briefing and verify required PPE.",
        "Inspect tools, equipment, access routes and work areas.",
        "Confirm approved drawings, materials and work permits.",
        "Record worker attendance and planned activities.",
        "Document progress, delays, incidents and material usage.",
    ]

    if request.weather == "rainy":
        checklist.append(
            "Check drainage, slippery surfaces, electrical hazards "
            "and excavation stability before work."
        )
    elif request.weather == "hot":
        checklist.append(
            "Provide drinking water, rest breaks, shade and heat-stress monitoring."
        )
    elif request.weather == "windy":
        checklist.append(
            "Assess lifting operations, scaffolding and loose materials; "
            "stop unsafe work."
        )

    if request.issues.strip().lower() not in {
        "none", "none reported", "no issues", "n/a"
    }:
        checklist.append(
            "Review and assign responsibility for reported issues: "
            + request.issues.strip()
        )

    return {
        "activity": "5.2",
        "date": date.today().isoformat(),
        "project_type": request.project_type,
        "workers": request.workers,
        "planned_work": request.work_planned,
        "checklist": checklist,
        "prototype": True,
        "note": "Confirm all actions against the approved site safety plan.",
    }


# Activity 5.3: Weather-Aware Construction Advisory

class WeatherRequest(BaseModel):
    temperature_c: float = Field(ge=-50, le=65)
    rain_expected: bool
    wind_kmh: float = Field(ge=0, le=300)
    lightning_expected: bool = False
    work_type: Literal[
        "general", "concrete", "excavation", "roofing", "lifting"
    ] = "general"


@router.post("/weather-advisory")
def weather_advisory(request: WeatherRequest):
    warnings = []
    recommendations = []

    if request.lightning_expected:
        warnings.append("Lightning risk")
        recommendations.append(
            "Suspend exposed outdoor work and follow the site's "
            "lightning emergency procedure."
        )

    if request.rain_expected:
        warnings.append("Rain expected")
        recommendations.append(
            "Protect materials, check drainage and assess slippery surfaces."
        )

        if request.work_type == "excavation":
            recommendations.append(
                "Have a competent person reassess excavation stability "
                "and water accumulation before entry."
            )

        if request.work_type == "concrete":
            recommendations.append(
                "Protect fresh concrete from rain and follow the approved "
                "placing and curing procedure."
            )

    if request.temperature_c >= 35:
        warnings.append("High temperature")
        recommendations.append(
            "Assess heat stress and provide water, shade and suitable rest breaks."
        )
    elif request.temperature_c <= 5:
        warnings.append("Low temperature")
        recommendations.append(
            "Check material-specific temperature requirements and cold-weather procedures."
        )

    if request.wind_kmh >= 40:
        warnings.append("Strong wind")
        recommendations.append(
            "Assess lifting, work at height and unsecured materials. "
            "Follow equipment-specific wind limits."
        )

    if request.work_type == "roofing" and (
        request.rain_expected or request.wind_kmh >= 30
    ):
        recommendations.append(
            "Pause roof work if surfaces or wind conditions make it unsafe."
        )

    if request.work_type == "lifting" and request.wind_kmh >= 25:
        recommendations.append(
            "Check the lifting equipment manufacturer's wind limits "
            "and the approved lifting plan."
        )

    if not warnings:
        warnings.append("No threshold-based warning from the supplied values")
        recommendations.append(
            "Complete normal site checks. These inputs do not guarantee safe conditions."
        )

    return {
        "activity": "5.3",
        "inputs": request.model_dump(),
        "warnings": warnings,
        "recommendations": recommendations,
        "live_weather": False,
        "prototype": True,
        "note": (
            "Weather is user-supplied, not a live forecast. "
            "Use official weather alerts and site procedures. "
            "These example thresholds are not universal safety limits."
        ),
    }
