# Student Performance Prediction System

A Machine Learning–based Student Performance Prediction System that identifies students who may be **at risk of poor academic performance** using academic, attendance, engagement, and study-related features.

The project includes:

- Data generation and preprocessing
- Feature engineering
- Machine Learning model training
- Model evaluation
- Student risk prediction
- FastAPI REST API
- Interactive Swagger API documentation
- Streamlit dashboard support
- Actionable recommendations for students

---

## 1. Project Objective

The main objective of this project is to predict whether a student is **at risk of poor academic performance** before the final examination.

The system analyzes factors such as:

- Attendance
- Previous academic performance
- Unit test scores
- Midterm scores
- Assignment completion
- Homework completion
- Quiz performance
- Study hours
- Class participation

The prediction can help teachers identify students who may need additional academic support.

---

## 2. Problem Statement

Educational institutions often identify struggling students only after their final examination results.

This project attempts to solve that problem using Machine Learning.

The system predicts:

> **Is this student at risk of poor academic performance?**

The problem is formulated as a **binary classification problem**:

```text
0 → Not At Risk
1 → At Risk




3. Machine Learning Approach
The project follows a complete Machine Learning pipeline:
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Train / Validation / Test Split
   ↓
Model Training
   ↓
Cross Validation
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Prediction
   ↓
FastAPI
   ↓
Dashboard

4. Features Used
The model uses the following input features:
Feature	Description
attendance	Student attendance percentage
previous_percentage	Previous academic percentage
unit_test_score	Unit test performance
midterm_score	Midterm examination score
assignment_percentage	Assignment completion percentage
homework_completion	Homework completion percentage
quiz_average	Average quiz score
study_hours	Average study hours per day
class_participation	Classroom participation percentage


Additional engineered features are created during preprocessing.
Engineered Features
academic_average
engagement_score
low_attendance
low_academic_score
overall_score

These features help the model capture relationships between academic performance and student engagement.
5. Machine Learning Models
The project experiments with multiple classification models.
Logistic Regression
Used as a baseline classification model.

Configuration includes:
- Feature scaling
- Class balancing
- Missing-value handling
Random Forest
The main model used in the project.

Configuration:
RandomForestClassifier(    n_estimators=300,    max_depth=10,    min_samples_leaf=3,    class_weight="balanced",    random_state=42,    n_jobs=-1)


Random Forest was selected because it can capture nonlinear relationships between student behavior and academic performance.
6. Model Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Stratified K-Fold Cross Validation
Why F1 Score?
In this project, identifying an at-risk student is important.

A model that simply predicts most students as "not at risk" could achieve high accuracy while failing to identify struggling students.

Therefore, Precision, Recall and F1 Score are important evaluation metrics.
7. Risk Classification
The prediction probability is converted into a risk level.
Probability >= 0.75
        ↓
   HIGH RISK

Probability >= 0.50
        ↓
  MEDIUM RISK

Probability < 0.50
        ↓
    LOW RISK

The thresholds can be modified depending on the school's requirements.
8. Project Structure
Student Performance Prediction System/
│
├── data/
│   ├── students_raw.csv
│   └── students_clean.csv
│
├── models/
│   └── student_risk_model.pkl
│
├── outputs/
│   ├── feature_importance.csv
│   ├── feature_importance.png
│   ├── predictions.csv
│   └── model_metrics.csv
│
├── src/
│   ├── generate_data.py
│   ├── prepare.py
│   ├── train.py
│   └── predict.py
│
├── api/
│   └── app.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── README.md
└── .gitignore

9. Installation
Step 1: Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>

Move into the project directory:
cd "Student Performance Prediction System"

Step 2: Create a Virtual Environment
Windows:
python -m venv venv

Activate it:
venv\Scripts\activate

You should see:
(venv)

in your terminal.
10. Install Dependencies
Install all required packages:
pip install -r requirements.txt

Current dependencies:
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
fastapi
uvicorn
streamlit
requests

11. Generate Dataset
Run:
python src/generate_data.py

This generates the raw student dataset:
data/students_raw.csv

The dataset contains student academic and engagement information.
12. Prepare the Data
Run:
python src/prepare.py

This performs:
- Duplicate removal
- Missing-value handling
- Numeric conversion
- Feature engineering
- Data validation
Output:
data/students_clean.csv

13. Train the Model
Run:
python src/train.py

The training pipeline:
1. Loads cleaned data
2. Separates features and target
3. Splits the dataset
4. Trains Logistic Regression
5. Trains Random Forest
6. Performs Stratified K-Fold Cross Validation
7. Calculates evaluation metrics
8. Selects the best model
9. Saves the trained model
The trained model is saved as:
models/student_risk_model.pkl

14. Run a Local Prediction
Run:
python src/predict.py

Example output:
==================================================
STUDENT PERFORMANCE PREDICTION
==================================================

Risk Status: AT RISK
Risk Probability: 99.81%
Risk Level: HIGH RISK

The exact probability depends on the trained model.
15. Run FastAPI
Start the API using:
uvicorn api.app:app --reload

Expected output:
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.

The API is now available at:
http://127.0.0.1:8000

16. API Endpoints
GET /
Checks whether the API is running.

Example response:
{
  "status": "ok",
  "service": "Student Performance Prediction API"
}

GET /health
Checks API and model status.

Example:
{
  "status": "healthy",
  "model_loaded": true
}

POST /predict
Predicts the risk level of a student.
Request
{
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

Example Response
{
  "at_risk": true,
  "risk_probability": 0.9981,
  "risk_percentage": 99.81,
  "risk_level": "HIGH RISK",
  "recommendations": [
    "Improve attendance.",
    "Focus on weak academic subjects.",
    "Increase study hours.",
    "Improve homework and assignment completion.",
    "Increase classroom participation."
  ]
}

The exact prediction depends on the trained model.
17. Swagger API Documentation
FastAPI automatically provides interactive API documentation.

Open:
http://127.0.0.1:8000/docs

You can test the /predict endpoint directly from Swagger.
Example High-Performing Student
{
  "attendance": 100,
  "previous_percentage": 100,
  "unit_test_score": 100,
  "midterm_score": 100,
  "assignment_percentage": 100,
  "homework_completion": 100,
  "quiz_average": 100,
  "study_hours": 24,
  "class_participation": 100
}

This should normally produce a LOW RISK prediction.

The exact probability is determined by the trained model.
18. Example Student Scenarios
Scenario 1 — High Performing Student
Attendance: 95%
Previous Percentage: 92%
Unit Test: 90%
Midterm: 94%
Assignments: 95%
Homework: 96%
Quiz Average: 91%
Study Hours: 5
Participation: 90%

Expected:
LOW RISK

Scenario 2 — Average Student
Attendance: 75%
Previous Percentage: 68%
Unit Test: 65%
Midterm: 62%
Assignments: 70%
Homework: 72%
Quiz Average: 66%
Study Hours: 2.5
Participation: 65%

Expected:
LOW / MEDIUM RISK

Scenario 3 — Struggling Student
Attendance: 55%
Previous Percentage: 42%
Unit Test: 38%
Midterm: 40%
Assignments: 45%
Homework: 40%
Quiz Average: 35%
Study Hours: 1
Participation: 30%

Expected:
HIGH RISK

19. Streamlit Dashboard
The project is designed to support a teacher-friendly Streamlit dashboard.

Run:
streamlit run dashboard/app.py

The dashboard can provide:
Student Information
       ↓
Performance Input
       ↓
FastAPI Prediction
       ↓
Risk Probability
       ↓
Risk Level
       ↓
Recommendations

The dashboard is intended to make the ML model usable by teachers without requiring them to interact directly with the API.
20. Dashboard Features
The planned dashboard includes:
Student Input
Teachers can enter:
- Attendance
- Previous marks
- Unit test marks
- Midterm marks
- Assignment completion
- Homework completion
- Quiz average
- Study hours
- Class participation
Prediction
The dashboard displays:
Risk Probability
Risk Level
At-Risk Status

Recommendations
The system provides targeted suggestions such as:
Improve attendance
Increase study hours
Improve assignment completion
Focus on weak academic areas
Increase classroom participation

21. Data Leakage Consideration
A major concern in student performance prediction is data leakage.

Features that are only available after the final examination should not be used to predict whether a student will fail before the final examination.

For example, using:
Final Exam Score
Final Grade
Final Result

as input features would create target leakage if the objective is to predict final performance before the examination.

Therefore, this project focuses on information that can be available before the final result.
22. Important Note About the Dataset
The initial version of this project uses a synthetically generated dataset for development and demonstration.

Therefore:
- Model probabilities are not real-world probabilities.
- Performance metrics should not be interpreted as real school performance.
- Feature importance represents predictive association, not causation.
- The model should not be used for high-stakes educational decisions without validation on real institutional data.
For production use, the model should be retrained using an appropriate real-world educational dataset.
23. Future Improvements
Potential improvements include:
Machine Learning
- XGBoost
- LightGBM
- Gradient Boosting
- Hyperparameter optimization
- Probability calibration
- Threshold optimization
- SHAP explainability
Data
- Real student datasets
- Historical academic records
- Subject-wise performance
- Monthly attendance trends
- Assignment submission history
- Behavioral indicators
API
- Authentication
- Request logging
- Rate limiting
- Model versioning
- Prediction history
- Database integration
Dashboard
- Student search
- Class-level risk distribution
- Risk trend over time
- Student comparison
- Teacher recommendations
- Downloadable reports
- CSV batch prediction
Deployment
Possible deployment options:
Docker
Render
Railway
AWS
Azure
Google Cloud

24. Business / Real-World Impact
The system can help educational institutions move from:
Reactive Education
        ↓
Student fails
        ↓
Teacher identifies problem

to:
Predictive Education
        ↓
Identify At-Risk Student
        ↓
Teacher Intervention
        ↓
Academic Support
        ↓
Improved Performance

The goal is not to replace teachers.

The goal is to provide teachers with an additional data-driven early warning system.
25. Technologies Used
Technology	Purpose
Python	Core programming language
Pandas	Data manipulation
NumPy	Numerical computation
Scikit-learn	Machine Learning
Matplotlib	Visualization
Seaborn	Data visualization
Joblib	Model serialization
FastAPI	REST API
Uvicorn	API server
Streamlit	Dashboard
Git	Version control
GitHub	Code hosting


26. Skills Demonstrated
This project demonstrates practical knowledge of:
- Python
- Pandas
- NumPy
- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Classification
- Logistic Regression
- Random Forest
- Cross Validation
- Stratified K-Fold
- Class Imbalance
- Model Evaluation
- F1 Score
- Precision
- Recall
- ROC-AUC
- Feature Importance
- Model Serialization
- FastAPI
- REST APIs
- Swagger
- Streamlit
- Git/GitHub
- ML Deployment Architecture
27. Author
Ratan Kumar Soni

Data Science / Machine Learning Enthusiast

Skills:
Python
Pandas
SQL
Machine Learning
FastAPI
Data Analysis
Excel
AI Tools

28. Disclaimer
This project is intended for educational, demonstration, and portfolio purposes.

A student's predicted risk should not be treated as a definitive judgment about their academic ability.

Any real-world deployment should include:
- Human review
- Data privacy controls
- Bias evaluation
- Model validation
- Appropriate consent and governance
29. License
This project is available for educational and portfolio purposes.

Add an appropriate open-source license before public distribution if required.