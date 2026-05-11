"""
Introduce realistic data quality issues into Smartphone_Dataset.csv
to demonstrate data preparation/wrangling/cleaning skills.
"""
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('Smartphone_Dataset.csv')
original_shape = df.shape
print(f"Original dataset: {original_shape[0]} rows, {original_shape[1]} columns")

# ============================================================
# 1. MISSING VALUES - ~5% across different columns
# ============================================================
print("\n--- Introducing Missing Values ---")

# Age: 3% missing
age_missing_idx = df.sample(n=int(len(df)*0.03), random_state=42).index
df.loc[age_missing_idx, 'age'] = np.nan
print(f"Age: {len(age_missing_idx)} missing values")

# Sleep hours: 4% missing
sleep_missing_idx = df.sample(n=int(len(df)*0.04), random_state=43).index
df.loc[sleep_missing_idx, 'sleep_hours'] = np.nan
print(f"Sleep hours: {len(sleep_missing_idx)} missing values")

# Social media hours: 2% missing
social_missing_idx = df.sample(n=int(len(df)*0.02), random_state=44).index
df.loc[social_missing_idx, 'social_media_hours'] = np.nan
print(f"Social media hours: {len(social_missing_idx)} missing values")

# Notifications: 3% missing
notif_missing_idx = df.sample(n=int(len(df)*0.03), random_state=45).index
df.loc[notif_missing_idx, 'notifications_per_day'] = np.nan
print(f"Notifications: {len(notif_missing_idx)} missing values")

# Weekend screen time: 2% missing
weekend_missing_idx = df.sample(n=int(len(df)*0.02), random_state=46).index
df.loc[weekend_missing_idx, 'weekend_screen_time'] = np.nan
print(f"Weekend screen time: {len(weekend_missing_idx)} missing values")

# Stress level: 3% missing
stress_missing_idx = df.sample(n=int(len(df)*0.03), random_state=47).index
df.loc[stress_missing_idx, 'stress_level'] = np.nan
print(f"Stress level: {len(stress_missing_idx)} missing values")

# Gender: 2% missing
gender_missing_idx = df.sample(n=int(len(df)*0.02), random_state=48).index
df.loc[gender_missing_idx, 'gender'] = np.nan
print(f"Gender: {len(gender_missing_idx)} missing values")

# Addiction level: 5% missing
addiction_missing_idx = df.sample(n=int(len(df)*0.05), random_state=49).index
df.loc[addiction_missing_idx, 'addiction_level'] = np.nan
print(f"Addiction level: {len(addiction_missing_idx)} missing values")

# ============================================================
# 2. DUPLICATE RECORDS - ~1.5% duplicates
# ============================================================
print("\n--- Introducing Duplicate Records ---")
n_duplicates = int(len(df) * 0.015)
duplicate_indices = df.sample(n=n_duplicates, random_state=50).index
duplicates = df.loc[duplicate_indices].copy()
# Slightly modify some duplicates to make them harder to detect
for i, idx in enumerate(duplicate_indices[:len(duplicate_indices)//2]):
    if pd.notna(duplicates.loc[idx, 'notifications_per_day']):
        duplicates.loc[idx, 'notifications_per_day'] = duplicates.loc[idx, 'notifications_per_day'] + np.random.randint(-2, 3)
df = pd.concat([df, duplicates], ignore_index=True)
print(f"Added {n_duplicates} duplicate records")

# ============================================================
# 3. OUTLIERS / UNREALISTIC VALUES
# ============================================================
print("\n--- Introducing Outliers and Unrealistic Values ---")

# Extreme ages (negative and very high)
outlier_age_idx = df.sample(n=15, random_state=51).index
df.loc[outlier_age_idx[:5], 'age'] = np.random.choice([-5, -3, -1, 0, 150], size=5)
df.loc[outlier_age_idx[5:10], 'age'] = np.random.choice([200, 999, 150, 180, 300], size=5)
df.loc[outlier_age_idx[10:], 'age'] = np.random.choice([8, 9, 10, 11, 12], size=5)
print(f"Age: 15 unrealistic values (negative, >100, <13)")

# Extreme screen time (>24 hours)
outlier_screen_idx = df.sample(n=20, random_state=52).index
df.loc[outlier_screen_idx, 'daily_screen_time_hours'] = np.random.uniform(25, 48, size=20).round(2)
print(f"Daily screen time: 20 values >24 hours")

# Negative sleep hours
outlier_sleep_idx = df.sample(n=10, random_state=53).index
df.loc[outlier_sleep_idx, 'sleep_hours'] = np.random.choice([-3, -2, -1, -5, -8], size=10)
print(f"Sleep hours: 10 negative values")

# Extreme sleep hours (>24)
outlier_sleep2_idx = df.sample(n=8, random_state=54).index
df.loc[outlier_sleep2_idx, 'sleep_hours'] = np.random.uniform(25, 40, size=8).round(2)
print(f"Sleep hours: 8 values >24 hours")

# Negative notifications
outlier_notif_idx = df.sample(n=12, random_state=55).index
df.loc[outlier_notif_idx, 'notifications_per_day'] = np.random.randint(-500, -1, size=12)
print(f"Notifications: 12 negative values")

# Extreme weekend screen time
outlier_weekend_idx = df.sample(n=10, random_state=56).index
df.loc[outlier_weekend_idx, 'weekend_screen_time'] = np.random.uniform(25, 50, size=10).round(2)
print(f"Weekend screen time: 10 values >24 hours")

# Negative social media hours
outlier_social_idx = df.sample(n=8, random_state=57).index
df.loc[outlier_social_idx, 'social_media_hours'] = np.random.uniform(-5, -0.5, size=8).round(2)
print(f"Social media hours: 8 negative values")

# Negative gaming hours
outlier_gaming_idx = df.sample(n=6, random_state=58).index
df.loc[outlier_gaming_idx, 'gaming_hours'] = np.random.uniform(-3, -0.1, size=6).round(2)
print(f"Gaming hours: 6 negative values")

# ============================================================
# 4. INCONSISTENT CATEGORICAL VALUES (typos, variations)
# ============================================================
print("\n--- Introducing Inconsistent Categorical Values ---")

# Gender typos
gender_typo_idx = df[df['gender'].notna()].sample(n=30, random_state=59).index
gender_typos = ['male', 'MALE', 'M', 'female', 'FEMALE', 'F', 'other', 'OTHER', 'O', 'm', 'f', 'o']
df.loc[gender_typo_idx, 'gender'] = np.random.choice(gender_typos, size=30)
print(f"Gender: 30 inconsistent values (typos, case variations)")

# Stress level typos
stress_typo_idx = df[df['stress_level'].notna()].sample(n=25, random_state=60).index
stress_typos = ['low', 'LOW', 'L', 'medium', 'MED', 'Med', 'high', 'HIGH', 'H', 'hi', 'lo', 'med']
df.loc[stress_typo_idx, 'stress_level'] = np.random.choice(stress_typos, size=25)
print(f"Stress level: 25 inconsistent values (typos, case variations)")

# Academic work impact typos
impact_typo_idx = df[df['academic_work_impact'].notna()].sample(n=20, random_state=61).index
impact_typos = ['no', 'NO', 'N', 'yes', 'YES', 'Y', 'Nope', 'Yeah', 'n', 'y']
df.loc[impact_typo_idx, 'academic_work_impact'] = np.random.choice(impact_typos, size=20)
print(f"Academic work impact: 20 inconsistent values (typos, case variations)")

# Addiction level typos
addiction_typo_idx = df[df['addiction_level'].notna()].sample(n=20, random_state=62).index
addiction_typos = ['none', 'NONE', 'mild', 'MILD', 'moderate', 'MOD', 'severe', 'SEV', 'high', 'extreme']
df.loc[addiction_typo_idx, 'addiction_level'] = np.random.choice(addiction_typos, size=20)
print(f"Addiction level: 20 inconsistent values (typos, case variations)")

# ============================================================
# 5. INCONSISTENT LABELS (addicted_label vs addiction_level mismatch)
# ============================================================
print("\n--- Introducing Label Inconsistencies ---")

# Some records where addicted_label=0 but addiction_level=Moderate/Severe
mismatch_idx = df[(df['addicted_label'] == 0) & (df['addiction_level'].notna())].sample(n=30, random_state=63).index
df.loc[mismatch_idx, 'addiction_level'] = np.random.choice(['Moderate', 'Severe'], size=30)
print(f"Label mismatches: 30 records with addicted_label=0 but level=Moderate/Severe")

# Some records where addicted_label=1 but addiction_level=None/mild
mismatch_idx2 = df[(df['addicted_label'] == 1) & (df['addiction_level'].notna())].sample(n=25, random_state=64).index
df.loc[mismatch_idx2, 'addiction_level'] = np.random.choice(['None', 'Mild'], size=25)
print(f"Label mismatches: 25 records with addicted_label=1 but level=None/Mild")

# ============================================================
# 6. DATA TYPE ISSUES - some numeric values stored as strings
# ============================================================
print("\n--- Introducing Data Type Issues ---")

# Some ages stored as strings
age_str_idx = df[df['age'].notna()].sample(n=20, random_state=65).index
df.loc[age_str_idx, 'age'] = df.loc[age_str_idx, 'age'].apply(lambda x: f"{int(x)}" if pd.notna(x) else x)
print(f"Age: 20 values stored as strings")

# ============================================================
# SAVE THE DIRTY DATASET
# ============================================================
df.to_csv('Smartphone_Dataset.csv', index=False)
print(f"\n{'='*60}")
print(f"Dirty dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Added {df.shape[0] - original_shape[0]} rows (duplicates)")
print(f"{'='*60}")

# Print summary of issues
print("\n=== DATA QUALITY ISSUES INTRODUCED ===")
print("1. Missing values across 8 columns (2-5% each)")
print("2. ~1.5% duplicate records")
print("3. Outliers/unrealistic values:")
print("   - Negative ages, ages >100, ages <13")
print("   - Screen time >24 hours")
print("   - Negative sleep hours, sleep >24 hours")
print("   - Negative notifications")
print("   - Negative social media/gaming hours")
print("4. Inconsistent categorical values (typos, case variations)")
print("5. Label inconsistencies (addicted_label vs addiction_level)")
print("6. Data type issues (numeric values as strings)")
