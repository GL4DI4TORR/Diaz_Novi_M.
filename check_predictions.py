import pandas as pd
from prediction.ml_service import ml_service

# Load a few samples from the dataset
df = pd.read_csv('Smartphone_Dataset.csv')
samples = df.head(10)

print('Sample predictions from ML model:')
for idx, row in samples.iterrows():
    data = {
        'age': int(row['age']) if pd.notna(row['age']) else 25,
        'gender': str(row['gender']).strip() if pd.notna(row['gender']) else 'Other',
        'daily_screen_time_hours': float(row['daily_screen_time_hours']) if pd.notna(row['daily_screen_time_hours']) else 5.0,
        'social_media_hours': float(row['social_media_hours']) if pd.notna(row['social_media_hours']) else 2.0,
        'gaming_hours': float(row['gaming_hours']) if pd.notna(row['gaming_hours']) else 1.0,
        'work_study_hours': float(row['work_study_hours']) if pd.notna(row['work_study_hours']) else 3.0,
        'sleep_hours': float(row['sleep_hours']) if pd.notna(row['sleep_hours']) else 7.0,
        'notifications_per_day': int(row['notifications_per_day']) if pd.notna(row['notifications_per_day']) else 100,
        'app_opens_per_day': int(row['app_opens_per_day']) if pd.notna(row['app_opens_per_day']) else 50,
        'weekend_screen_time': float(row['weekend_screen_time']) if pd.notna(row['weekend_screen_time']) else 6.0,
        'stress_level': str(row['stress_level']).strip() if pd.notna(row['stress_level']) else 'Medium',
        'academic_work_impact': str(row['academic_work_impact']).strip() if pd.notna(row['academic_work_impact']) else 'No'
    }

    result = ml_service.predict(data)
    print(f'Row {idx}: addicted_label={row["addicted_label"]}, addiction_level="{row["addiction_level"]}", ML prediction: {result["risk_level"]}')