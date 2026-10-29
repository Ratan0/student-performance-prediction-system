import joblib
import pandas as pd


MODEL_PATH = "models/student_risk_model.pkl"


FEATURES = [
    "attendance",
    "previous_percentage",
    "unit_test_score",
    "midterm_score",
    "assignment_percentage",
    "homework_completion",
    "quiz_average",
    "study_hours",
    "class_participation"
]


def create_features(student):

    student["academic_average"] = (
        student["unit_test_score"]
        + student["midterm_score"]
        + student["quiz_average"]
    ) / 3

    student["engagement_score"] = (
        student["assignment_percentage"]
        + student["homework_completion"]
        + student["class_participation"]
    ) / 3

    student["low_attendance"] = int(
        student["attendance"] < 75
    )

    student["low_academic_score"] = int(
        student["academic_average"] < 50
    )

    student["overall_score"] = (
        student["previous_percentage"] * 0.25
        + student["academic_average"] * 0.40
        + student["engagement_score"] * 0.20
        + student["attendance"] * 0.15
    )

    return student


def get_risk_level(probability):

    if probability >= 0.75:

        return "HIGH RISK"

    elif probability >= 0.50:

        return "MEDIUM RISK"

    else:

        return "LOW RISK"


def get_recommendations(student):

    recommendations = []

    if student["attendance"] < 75:

        recommendations.append(
            "Improve attendance."
        )

    if student["previous_percentage"] < 50:

        recommendations.append(
            "Provide academic mentoring."
        )

    if student["unit_test_score"] < 50:

        recommendations.append(
            "Provide additional unit-test practice."
        )

    if student["midterm_score"] < 50:

        recommendations.append(
            "Provide targeted midterm revision."
        )

    if student["assignment_percentage"] < 70:

        recommendations.append(
            "Monitor assignment completion."
        )

    if student["homework_completion"] < 70:

        recommendations.append(
            "Improve homework completion."
        )

    if student["study_hours"] < 2:

        recommendations.append(
            "Increase daily study time."
        )

    if student["class_participation"] < 50:

        recommendations.append(
            "Encourage classroom participation."
        )

    return recommendations


def predict_student(student):

    model = joblib.load(MODEL_PATH)

    student = create_features(student)

    row = pd.DataFrame(
        [student]
    )

    # Same feature order as training
    model_features = FEATURES + [
        "academic_average",
        "engagement_score",
        "low_attendance",
        "low_academic_score",
        "overall_score"
    ]

    row = row[model_features]

    probability = model.predict_proba(
        row
    )[0][1]

    prediction = int(
        probability >= 0.5
    )

    risk_level = get_risk_level(
        probability
    )

    recommendations = get_recommendations(
        student
    )

    return {
        "at_risk": prediction,
        "risk_probability": round(
            probability,
            4
        ),
        "risk_level": risk_level,
        "recommendations": recommendations
    }


if __name__ == "__main__":

    student = {

        "attendance": 62,

        "previous_percentage": 48,

        "unit_test_score": 43,

        "midterm_score": 41,

        "assignment_percentage": 55,

        "homework_completion": 50,

        "quiz_average": 45,

        "study_hours": 1.5,

        "class_participation": 40
    }

    result = predict_student(
        student
    )

    print("\n")
    print("=" * 50)
    print("STUDENT PERFORMANCE PREDICTION")
    print("=" * 50)

    print(
        "Risk Status:",
        "AT RISK"
        if result["at_risk"]
        else "NOT AT RISK"
    )

    print(
        "Risk Probability:",
        f"{result['risk_probability']:.2%}"
    )

    print(
        "Risk Level:",
        result["risk_level"]
    )

    print("\nRecommendations:")

    for item in result["recommendations"]:

        print(" -", item)