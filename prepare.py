import os
import pandas as pd


INPUT_FILE = "data/students_raw.csv"
OUTPUT_FILE = "data/students_clean.csv"


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


def load_data(path):

    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    return df


def clean_data(df):

    df = df.copy()

    # Remove duplicate students
    df = df.drop_duplicates(
        subset=["student_id"]
    )

    # Numeric columns
    numeric_columns = FEATURES + ["at_risk"]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Fill missing values using median
    for column in FEATURES:

        df[column] = df[column].fillna(
            df[column].median()
        )

    # Remove impossible values
    df["attendance"] = df["attendance"].clip(0, 100)

    df["previous_percentage"] = (
        df["previous_percentage"].clip(0, 100)
    )

    df["unit_test_score"] = (
        df["unit_test_score"].clip(0, 100)
    )

    df["midterm_score"] = (
        df["midterm_score"].clip(0, 100)
    )

    df["assignment_percentage"] = (
        df["assignment_percentage"].clip(0, 100)
    )

    df["homework_completion"] = (
        df["homework_completion"].clip(0, 100)
    )

    df["quiz_average"] = (
        df["quiz_average"].clip(0, 100)
    )

    df["class_participation"] = (
        df["class_participation"].clip(0, 100)
    )

    df["study_hours"] = df["study_hours"].clip(0, 24)

    return df


def feature_engineering(df):

    df = df.copy()

    # Academic average
    df["academic_average"] = (
        df["unit_test_score"]
        + df["midterm_score"]
        + df["quiz_average"]
    ) / 3

    # Engagement score
    df["engagement_score"] = (
        df["assignment_percentage"]
        + df["homework_completion"]
        + df["class_participation"]
    ) / 3

    # Attendance warning
    df["low_attendance"] = (
        df["attendance"] < 75
    ).astype(int)

    # Academic warning
    df["low_academic_score"] = (
        df["academic_average"] < 50
    ).astype(int)

    # Overall performance indicator
    df["overall_score"] = (
        df["previous_percentage"] * 0.25
        + df["academic_average"] * 0.40
        + df["engagement_score"] * 0.20
        + df["attendance"] * 0.15
    )

    return df


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    df = load_data(INPUT_FILE)

    df = clean_data(df)

    df = feature_engineering(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSaved:", OUTPUT_FILE)