import os
import numpy as np
import pandas as pd


np.random.seed(42)

N_STUDENTS = 2000


def generate_data(n=N_STUDENTS):

    student_ids = [
        f"S{str(i).zfill(5)}"
        for i in range(1, n + 1)
    ]

    attendance = np.clip(
        np.random.normal(78, 14, n),
        40,
        100
    )

    previous_percentage = np.clip(
        np.random.normal(65, 15, n),
        25,
        100
    )

    unit_test_score = np.clip(
        np.random.normal(62, 18, n),
        15,
        100
    )

    midterm_score = np.clip(
        np.random.normal(64, 18, n),
        15,
        100
    )

    assignment_percentage = np.clip(
        np.random.normal(72, 18, n),
        20,
        100
    )

    homework_completion = np.clip(
        np.random.normal(70, 20, n),
        15,
        100
    )

    quiz_average = np.clip(
        np.random.normal(65, 18, n),
        15,
        100
    )

    study_hours = np.clip(
        np.random.normal(2.5, 1.2, n),
        0.2,
        8
    )

    class_participation = np.clip(
        np.random.normal(65, 20, n),
        10,
        100
    )

    # Hidden relationship used only to generate realistic outcomes
    risk_score = (
        (100 - attendance) * 0.20
        + (100 - previous_percentage) * 0.20
        + (100 - unit_test_score) * 0.15
        + (100 - midterm_score) * 0.20
        + (100 - assignment_percentage) * 0.08
        + (100 - homework_completion) * 0.05
        + (100 - quiz_average) * 0.07
        + (8 - study_hours) * 1.5
        + (100 - class_participation) * 0.05
    )

    # Add randomness
    risk_score += np.random.normal(0, 5, n)

    # Convert risk score to binary target
    at_risk = (risk_score > 45).astype(int)

    df = pd.DataFrame({
        "student_id": student_ids,
        "attendance": np.round(attendance, 2),
        "previous_percentage": np.round(previous_percentage, 2),
        "unit_test_score": np.round(unit_test_score, 2),
        "midterm_score": np.round(midterm_score, 2),
        "assignment_percentage": np.round(assignment_percentage, 2),
        "homework_completion": np.round(homework_completion, 2),
        "quiz_average": np.round(quiz_average, 2),
        "study_hours": np.round(study_hours, 2),
        "class_participation": np.round(class_participation, 2),
        "at_risk": at_risk
    })

    return df


if __name__ == "__main__":

    os.makedirs("data", exist_ok=True)

    df = generate_data()

    # Add a few duplicate rows intentionally
    duplicates = df.sample(10, random_state=42)

    df = pd.concat(
        [df, duplicates],
        ignore_index=True
    )

    # Add some missing values intentionally
    missing_indices = np.random.choice(
        df.index,
        size=20,
        replace=False
    )

    df.loc[
        missing_indices[:10],
        "attendance"
    ] = np.nan

    df.loc[
        missing_indices[10:],
        "midterm_score"
    ] = np.nan

    df.to_csv(
        "data/students_raw.csv",
        index=False
    )

    print("Dataset generated successfully.")
    print("Shape:", df.shape)
    print("\nTarget distribution:")
    print(df["at_risk"].value_counts())