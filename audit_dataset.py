import pandas as pd
import numpy as np

df = pd.read_csv('Smartphone_Dataset.csv')

print("=" * 60)
print("DATASET QUALITY AUDIT")
print("=" * 60)

print(f"\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for missing values
print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "NO MISSING VALUES")

# Check for duplicates
print(f"\n--- DUPLICATES ---")
print(f"Duplicate rows: {df.duplicated().sum()}")

# Check for unrealistic values
print("\n--- UNREALISTIC VALUES ---")
print(f"Negative ages: {(df['age'] < 0).sum()}")
print(f"Ages > 100: {(df['age'] > 100).sum()}")
print(f"Screen time > 24h: {(df['daily_screen_time_hours'] > 24).sum()}")
print(f"Sleep > 24h: {(df['sleep_hours'] > 24).sum()}")
print(f"Negative sleep: {(df['sleep_hours'] < 0).sum()}")
print(f"Negative notifications: {(df['notifications_per_day'] < 0).sum()}")

# Check for outliers
print("\n--- OUTLIERS (IQR method) ---")
numeric_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
               'work_study_hours', 'sleep_hours', 'notifications_per_day',
               'app_opens_per_day', 'weekend_screen_time']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if outliers > 0:
        print(f"{col}: {outliers} outliers")

# Check data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# Check unique values in categorical columns
print("\n--- CATEGORICAL VALUES ---")
for col in ['gender', 'stress_level', 'academic_work_impact', 'addiction_level']:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")

# Check for inconsistent data
print("\n--- DATA CONSISTENCY ---")
print(f"addicted_label values: {sorted(df['addicted_label'].unique())}")
print(f"addiction_level values: {df['addiction_level'].unique()}")
# Check if addiction_level matches addicted_label
mismatch = df[(df['addicted_label'] == 0) & (df['addiction_level'].isin(['Moderate', 'Severe']))].shape[0]
print(f"Label mismatches (addicted_label=0 but level=Moderate/Severe): {mismatch}")

print("\n--- SUMMARY ---")
print("Is the dataset CLEAN (no issues to demonstrate)?")
issues = []
if missing.sum() == 0: issues.append("No missing values")
if df.duplicated().sum() == 0: issues.append("No duplicates")
if (df['age'] < 0).sum() == 0 and (df['age'] > 100).sum() == 0: issues.append("No unrealistic ages")
if (df['daily_screen_time_hours'] > 24).sum() == 0: issues.append("No unrealistic screen time")
if (df['sleep_hours'] > 24).sum() == 0: issues.append("No unrealistic sleep hours")

for issue in issues:
    print(f"  - {issue}")

if len(issues) >= 3:
    print("\n*** DATASET IS TOO CLEAN - NEEDS DATA QUALITY ISSUES FOR DEMONSTRATION ***")
else:
    print("\n*** DATASET HAS ENOUGH ISSUES FOR DEMONSTRATION ***")
