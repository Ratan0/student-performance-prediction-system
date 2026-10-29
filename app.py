from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="Student Performance Prediction API",
    description="API for predicting students at risk of academic failure.",
    version="1.0.0"
)


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = "models/student_risk_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Input Schema
# --------------------------------------------------

class StudentInput(BaseModel):

    attendance: float = Field(
        ...,
        ge=0,
        le=100
    )

    previous_percentage: float = Field(
        ...,
        ge=0,
        le=100
    )

    unit_test_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    midterm_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    assignment_percentage: float = Field(
        ...,
        ge=0,
        le=100
    )

    homework_completion: float = Field(
        ...,
        ge=0,
        le=100
    )

    quiz_average: float = Field(
        ...,
        ge=0,
        le=100
    )

    study_hours: float = Field(
        ...,
        ge=0,
        le=24
    )

    class_participation: float = Field(
        ...,
        ge=0,
        le=100
    )


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def create_features(data):

    data["academic_average"] = (
        data["unit_test_score"]
        + data["midterm_score"]
        + data["quiz_average"]
    ) / 3

    data["engagement_score"] = (
        data["assignment_percentage"]
        + data["homework_completion"]
        + data["class_participation"]
    ) / 3

    data["low_attendance"] = (
        data["attendance"] < 75
    ).astype(int)

    data["low_academic_score"] = (
        data["academic_average"] < 50
    ).astype(int)

    data["overall_score"] = (
        data["previous_percentage"] * 0.25
        + data["academic_average"] * 0.40
        + data["engagement_score"] * 0.20
        + data["attendance"] * 0.15
    )

    return data


# --------------------------------------------------
# Risk Level
# --------------------------------------------------

def get_risk_level(probability):

    if probability >= 0.75:
        return "HIGH"

    elif probability >= 0.50:
        return "MEDIUM"

    return "LOW"


# --------------------------------------------------
# Recommendations
# --------------------------------------------------

def get_recommendations(data):

    recommendations = []

    if data["attendance"] < 75:
        recommendations.append(
            "Improve attendance"
        )

    if data["previous_percentage"] < 50:
        recommendations.append(
            "Provide academic mentoring"
        )

    if data["unit_test_score"] < 50:
        recommendations.append(
            "Provide additional unit-test practice"
        )

    if data["midterm_score"] < 50:
        recommendations.append(
            "Provide targeted midterm revision"
        )

    if data["assignment_percentage"] < 70:
        recommendations.append(
            "Monitor assignment completion"
        )

    if data["homework_completion"] < 70:
        recommendations.append(
            "Improve homework completion"
        )

    if data["study_hours"] < 2:
        recommendations.append(
            "Increase daily study time"
        )

    if data["class_participation"] < 50:
        recommendations.append(
            "Encourage classroom participation"
        )

    return recommendations


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "Student Performance Prediction API"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(student: StudentInput):

    try:

        data = student.model_dump()

        df = pd.DataFrame([data])

        df = create_features(df)

        features = [
            "attendance",
            "previous_percentage",
            "unit_test_score",
            "midterm_score",
            "assignment_percentage",
            "homework_completion",
            "quiz_average",
            "study_hours",
            "class_participation",
            "academic_average",
            "engagement_score",
            "low_attendance",
            "low_academic_score",
            "overall_score"
        ]

        df = df[features]

        probability = float(
            model.predict_proba(df)[0][1]
        )

        prediction = int(
            probability >= 0.50
        )

        risk_level = get_risk_level(
            probability
        )

        recommendations = get_recommendations(
            data
        )

        return {
            "at_risk": bool(prediction),
            "risk_probability": round(
                probability,
                4
            ),
            "risk_percentage": round(
                probability * 100,
                2
            ),
            "risk_level": risk_level,
            "recommendations": recommendations
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )