"""
QUICK REFERENCE GUIDE - Smartphone Addiction Detection System
==============================================================

This is a quick lookup guide for the project structure and components.
"""

QUICK_START = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                          QUICK START GUIDE                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝

1. SETUP THE PROJECT
   $ python -m venv venv
   $ venv\\Scripts\\activate
   $ pip install -r requirements.txt

2. INITIALIZE DATABASE
   $ python manage.py migrate
   $ python manage.py runserver

3. LOAD SAMPLE DATA
   Visit: http://localhost:8000/load-dataset/
   (Loads 7,500+ records from Smartphone_Dataset.csv)

4. MAKE PREDICTIONS
   Visit: http://localhost:8000/predict/
   (Fill form → Submit → Get risk assessment)

5. VIEW RESULTS
   Visit: http://localhost:8000/dashboard/
   (See statistics and analytics)

═════════════════════════════════════════════════════════════════════════════
"""

ARCHITECTURE = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                         SYSTEM ARCHITECTURE                                 ║
╚═════════════════════════════════════════════════════════════════════════════╝

┌─ DATA LAYER ─────────────────────────────────────────────────────────────┐
│ • Smartphone_Dataset.csv (7,500 records, 16 features)                   │
│ • db.sqlite3 (AddictionRecord model for storing predictions)            │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ ML SERVICE LAYER ───────────────────────────────────────────────────────┐
│ • best_addiction_model.joblib (trained Gradient Boosting model)         │
│ • ml_service.py (preprocessing, prediction, risk classification)        │
│ • Performance: 81% accuracy, AUC: 0.87                                  │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ APPLICATION LAYER ──────────────────────────────────────────────────────┐
│ • Django Views (predict, dashboard, records)                            │
│ • REST API (DRF serializers and viewsets)                               │
│ • HTML Templates (forms, dashboards, results)                           │
└────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ PRESENTATION LAYER ────────────────────────────────────────────────────┐
│ • Web Interface (Bootstrap 5, Crispy Forms)                             │
│ • JSON API Responses                                                    │
│ • Interactive Dashboards                                               │
└────────────────────────────────────────────────────────────────────────────┘

═════════════════════════════════════════════════════════════════════════════
"""

DATA_FEATURES = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                          DATASET FEATURES                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝

DEMOGRAPHICS
  → age (13-80 years)
  → gender (Male, Female, Other)

USAGE PATTERNS
  → daily_screen_time_hours (0-24 hours) [STRONGEST PREDICTOR]
  → social_media_hours (0-12 hours)
  → gaming_hours (0-12 hours)
  → work_study_hours (0-12 hours)
  → weekend_screen_time (0-12 hours)

BEHAVIORAL
  → notifications_per_day (0-500)
  → app_opens_per_day (0-300)

LIFESTYLE
  → sleep_hours (0-12 hours) [NEGATIVE correlation with addiction]
  → stress_level (Low, Medium, High)

LIFE IMPACT
  → academic_work_impact (Yes, No)

TARGET VARIABLES
  → addicted_label (0 = Not Addicted, 1 = Addicted)
  → addiction_level (None, Mild, Moderate, Severe)

═════════════════════════════════════════════════════════════════════════════
"""

DATA_ISSUES = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                    DATA QUALITY ISSUES TACKLED                              ║
╚═════════════════════════════════════════════════════════════════════════════╝

ISSUE #1: MISSING VALUES (10.92% of records)
  Problem:  Age (3%), Sleep (4%), Social Media (2%), Notifications (3%), etc.
  Solution: Median imputation for numeric, Mode for categorical
  Result:   ✓ Zero missing values

ISSUE #2: DUPLICATE RECORDS (~115 rows, 1.5%)
  Problem:  Exact duplicates across all features
  Solution: Drop duplicates, keep first occurrence
  Result:   ✓ 7,500 unique records

ISSUE #3: OUTLIERS & UNREALISTIC VALUES
  Problem:  Negative ages, screen time > 24h, sleep > 12h, negative sleep
  Solution: IQR method with realistic bounds (age: 13-80, screen time: 0-24h)
  Result:   ✓ All values within logical ranges

ISSUE #4: DATA TYPE INCONSISTENCIES
  Problem:  Numeric values stored as strings
  Solution: Type conversion with error handling
  Result:   ✓ Consistent data types

ISSUE #5: CATEGORICAL INCONSISTENCIES
  Problem:  Mixed case values ("male", "MALE", "Male")
  Solution: Standardization and normalization
  Result:   ✓ Uniform categorical values

ISSUE #6: LABEL INCONSISTENCIES
  Problem:  Mismatch between addiction_level and binary label
  Solution: Reconciliation with mapping
  Result:   ✓ Synchronized target variables

═════════════════════════════════════════════════════════════════════════════
"""

ML_MODELS = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                       MACHINE LEARNING MODELS                               ║
╚═════════════════════════════════════════════════════════════════════════════╝

MODELS TESTED:
  1. Logistic Regression          72% accuracy (baseline)
  2. Random Forest                78% accuracy
  3. ★ GRADIENT BOOSTING ★        81% accuracy ← SELECTED
  4. XGBoost                      80% accuracy
  5. LightGBM                     79% accuracy

SELECTED MODEL: GRADIENT BOOSTING
  Parameters:
    • n_estimators: 100
    • learning_rate: 0.1
    • max_depth: 5
  
  Why Selected:
    ✓ Highest accuracy (81%)
    ✓ Captures complex patterns
    ✓ Handles imbalanced data
    ✓ Feature importance scores
    ✓ Good cross-validation results

PERFORMANCE METRICS:
  • Accuracy:   81%   (81/100 predictions correct)
  • Precision:  82%   (reliably identifies addicted users)
  • Recall:     79%   (catches 79% of actually addicted)
  • F1-Score:   0.80  (balanced metric)
  • AUC-ROC:    0.87  (excellent discrimination)

RISK CLASSIFICATION:
  Probability 0.00-0.25 → Risk: LOW           (No Addiction)
  Probability 0.25-0.50 → Risk: MEDIUM        (Mild)
  Probability 0.50-0.75 → Risk: HIGH          (Moderate)
  Probability 0.75-1.00 → Risk: CRITICAL      (Severe)

═════════════════════════════════════════════════════════════════════════════
"""

DJANGO_ROUTES = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                         DJANGO URL ROUTES                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝

WEB INTERFACE:
  GET  http://localhost:8000/                    Home page
  GET  http://localhost:8000/dashboard/          Statistics dashboard
  GET  http://localhost:8000/predict/            Prediction form
  POST http://localhost:8000/predict/            Submit prediction
  GET  http://localhost:8000/records/            Browse records
  GET  http://localhost:8000/add-record/         Add record form
  POST http://localhost:8000/load-dataset/       Bulk import CSV

REST API:
  GET    http://localhost:8000/api/records/              List records
  POST   http://localhost:8000/api/records/              Create record
  GET    http://localhost:8000/api/records/{id}/         Get details
  PUT    http://localhost:8000/api/records/{id}/         Update record
  DELETE http://localhost:8000/api/records/{id}/         Delete record

═════════════════════════════════════════════════════════════════════════════
"""

HELPER_SCRIPTS = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                        HELPER SCRIPTS                                       ║
╚═════════════════════════════════════════════════════════════════════════════╝

DATA ASSESSMENT:
  python audit_dataset.py
  → Checks for missing values, duplicates, outliers, data types
  → Shows data quality metrics and issues

DATA QUALITY TESTING:
  python introduce_data_issues.py
  → Adds realistic data quality issues to CSV for demonstration
  → Shows cleaning skills (demonstrating ability to handle problems)

FEATURE ANALYSIS:
  python analyze_weekend.py
  → Analyzes weekend vs weekday patterns
  → Compares usage behavior across time periods

PREDICTION VALIDATION:
  python check_predictions.py
  → Validates model predictions
  → Checks confidence levels and risk distribution

DJANGO UTILITIES:
  python manage.py migrate              Initialize database
  python manage.py runserver            Start development server
  python manage.py createsuperuser      Create admin user
  python manage.py shell                Django Python shell
  python test_id_reset.py               Reset database IDs

PROJECT EXPLANATION:
  python explain_project.py
  → Generates this comprehensive explanation
  → Performs system health checks
  → Shows formatted project overview

═════════════════════════════════════════════════════════════════════════════
"""

SKILLS_MAP = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                       SKILLS DEMONSTRATED                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝

DATA SCIENCE:
  ✓ Data loading and exploration          ✓ Statistical analysis
  ✓ Data cleaning and preprocessing       ✓ EDA with visualizations
  ✓ Feature engineering                   ✓ Model selection
  ✓ Hyperparameter tuning                 ✓ Model evaluation
  ✓ Cross-validation                      ✓ Handling imbalanced data
  ✓ Correlation analysis                  ✓ Outlier detection

WEB DEVELOPMENT:
  ✓ Django MVT architecture               ✓ Django REST Framework
  ✓ Database models and migrations        ✓ HTML templates
  ✓ Form handling and validation          ✓ Bootstrap 5 styling
  ✓ URL routing                           ✓ Admin interface
  ✓ API serialization                     ✓ Pagination

SOFTWARE ENGINEERING:
  ✓ Clean code architecture               ✓ Error handling
  ✓ Modular design                        ✓ Documentation
  ✓ Version control ready                 ✓ Testing
  ✓ Production-ready code                 ✓ Database constraints
  ✓ Performance optimization

═════════════════════════════════════════════════════════════════════════════
"""

KEY_FILES = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                         KEY PROJECT FILES                                   ║
╚═════════════════════════════════════════════════════════════════════════════╝

DOCUMENTATION:
  📄 PROJECT_EXPLANATION.md                Complete project documentation
  📄 ML_Pipeline_Presentation.md          ML methodology and findings
  📄 Dataset Overview.md                  Dataset information and context
  📄 QUICK_REFERENCE.txt                  This file

DATA & MODELS:
  📊 Smartphone_Dataset.csv               Raw dataset (7,500 records)
  🤖 best_addiction_model.joblib          Trained ML model (serialized)
  💾 db.sqlite3                           SQLite database

DJANGO APP:
  🐍 manage.py                            Django management script
  📁 addiction_detection/                 Main Django project
     ├── settings.py                      Configuration
     ├── urls.py                          URL routing
     └── wsgi.py, asgi.py                 Server configs
  
  📁 prediction/                          Main app
     ├── models.py                        Database model
     ├── views.py                         Request handlers
     ├── ml_service.py                    ML integration
     ├── serializers.py                   REST API
     ├── urls.py                          App routes
     └── migrations/                      Database migrations
  
  📁 templates/prediction/                HTML templates
  📁 static/                              CSS, JS, images

ANALYSIS & TESTING:
  🔍 audit_dataset.py                     Data quality audit
  🔧 introduce_data_issues.py             Add test issues
  📈 analyze_weekend.py                   Feature analysis
  ✅ check_predictions.py                 Prediction testing
  🧪 tests.py                             Unit tests

HELPERS:
  🚀 explain_project.py                   This explanation script
  🔧 test_id_reset.py                     Database utility

═════════════════════════════════════════════════════════════════════════════
"""

EXAMPLE_PREDICTION = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                        EXAMPLE PREDICTIONS                                  ║
╚═════════════════════════════════════════════════════════════════════════════╝

EXAMPLE 1: Low-Risk User
  Input:
    Age: 25 years
    Daily Screen Time: 4 hours/day
    Social Media: 1 hour/day
    Gaming: 0.5 hours/day
    Sleep: 8 hours/night
    Notifications: 30/day
    Stress Level: Low
  
  Output:
    Risk Level: LOW
    Addiction Probability: 0.18 (18%)
    Label: No Addiction
    Recommendation: Your usage pattern suggests healthy smartphone habits

EXAMPLE 2: High-Risk User
  Input:
    Age: 22 years
    Daily Screen Time: 11 hours/day
    Social Media: 5 hours/day
    Gaming: 4 hours/day
    Sleep: 5 hours/night
    Notifications: 250/day
    Stress Level: High
  
  Output:
    Risk Level: CRITICAL
    Addiction Probability: 0.88 (88%)
    Label: Severe Addiction
    Recommendation: Significant addiction risk. Consider setting daily limits
                    and scheduling device-free time blocks.

EXAMPLE 3: Moderate-Risk User
  Input:
    Age: 28 years
    Daily Screen Time: 7 hours/day
    Social Media: 2.5 hours/day
    Gaming: 1.5 hours/day
    Sleep: 7 hours/night
    Notifications: 120/day
    Stress Level: Medium
  
  Output:
    Risk Level: HIGH
    Addiction Probability: 0.62 (62%)
    Label: Moderate Addiction
    Recommendation: Consider reducing daily screen time and taking regular breaks.

═════════════════════════════════════════════════════════════════════════════
"""

RESOURCES = """
╔═════════════════════════════════════════════════════════════════════════════╗
║                           RESOURCES                                         ║
╚═════════════════════════════════════════════════════════════════════════════╝

DOCUMENTATION:
  • PROJECT_EXPLANATION.md        ← Read first! Comprehensive overview
  • ML_Pipeline_Presentation.md   ← Detailed ML methodology
  • Dataset Overview.md            ← Dataset documentation
  
SCRIPTS:
  • python explain_project.py      ← Run this for interactive explanation
  • python audit_dataset.py        ← Analyze data quality
  • python analyze_weekend.py      ← Analyze patterns

NOTEBOOKS:
  • ML_Notebook.ipynb              ← Main ML development
  • smartphone_addiction_ml_pipeline.ipynb
  • proj.ipynb

CONFIGURATION:
  • requirements.txt               ← Python dependencies

═════════════════════════════════════════════════════════════════════════════
"""

def main():
    """Display quick reference guide"""
    guides = [
        QUICK_START,
        ARCHITECTURE,
        DATA_FEATURES,
        DATA_ISSUES,
        ML_MODELS,
        DJANGO_ROUTES,
        HELPER_SCRIPTS,
        SKILLS_MAP,
        KEY_FILES,
        EXAMPLE_PREDICTION,
        RESOURCES
    ]
    
    for guide in guides:
        print(guide)
    
    print("""
╔═════════════════════════════════════════════════════════════════════════════╗
║                            END OF GUIDE                                     ║
║                                                                             ║
║  For detailed information, see: PROJECT_EXPLANATION.md                    ║
║  For interactive explanation, run: python explain_project.py              ║
╚═════════════════════════════════════════════════════════════════════════════╝
""")

if __name__ == '__main__':
    main()
