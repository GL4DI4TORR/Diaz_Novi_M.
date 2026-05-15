"""
Smartphone Addiction Dataset - FINAL COMPREHENSIVE CLEANING
===========================================================

Complete cleaning in one script with all improvements applied
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("FINAL COMPREHENSIVE DATA CLEANING - ALL ISSUES RESOLVED")
print("=" * 100)

# Load the ORIGINAL dataset
print("\n>>> LOADING ORIGINAL DATASET...")
df = pd.read_csv('Smartphone_Dataset_Original_Backup.csv')
original_shape = df.shape
print(f"✓ Loaded: {original_shape[0]:,} rows × {original_shape[1]} columns")

# ============================================================================
# STEP 1: REMOVE DUPLICATE RECORDS
# ============================================================================
print("\n>>> STEP 1: REMOVING DUPLICATES...")
duplicates_count = df.duplicated().sum()
df = df.drop_duplicates()
print(f"✓ Removed {duplicates_count} duplicate records")

# ============================================================================
# STEP 2: COMPREHENSIVE CATEGORICAL STANDARDIZATION (BEFORE filling NaN)
# ============================================================================
print("\n>>> STEP 2: STANDARDIZING CATEGORICAL VALUES...")

def standardize_string(val, mapping):
    """Convert value to standard format"""
    if pd.isna(val):
        return np.nan
    val_clean = str(val).strip().lower()
    return mapping.get(val_clean, np.nan)

# Gender
gender_mapping = {
    'm': 'Male', 'male': 'Male', 'male ': 'Male',
    'f': 'Female', 'female': 'Female', 'female ': 'Female',
    'o': 'Other', 'other': 'Other', 'other ': 'Other',
    '1': 'Male', '2': 'Female', '3': 'Other'
}
df['gender'] = df['gender'].apply(lambda x: standardize_string(x, gender_mapping) if pd.notna(x) else np.nan)
print("✓ Gender standardized")

# Stress Level
stress_mapping = {
    'l': 'Low', 'lo': 'Low', 'low': 'Low',
    'm': 'Medium', 'med': 'Medium', 'medium': 'Medium',
    'h': 'High', 'hi': 'High', 'high': 'High',
    '1': 'Low', '2': 'Medium', '3': 'High'
}
df['stress_level'] = df['stress_level'].apply(lambda x: standardize_string(x, stress_mapping) if pd.notna(x) else np.nan)
print("✓ Stress level standardized")

# Academic/Work Impact
impact_mapping = {
    'y': 'Yes', 'yes': 'Yes', 'yeah': 'Yes', 'yep': 'Yes',
    'n': 'No', 'no': 'No', 'nope': 'No', 'nah': 'No',
    '1': 'Yes', '0': 'No'
}
df['academic_work_impact'] = df['academic_work_impact'].apply(lambda x: standardize_string(x, impact_mapping) if pd.notna(x) else np.nan)
print("✓ Academic/Work impact standardized")

# Addiction Level
addiction_level_mapping = {
    'n': 'None', 'none': 'None',
    'm': 'Mild', 'mild': 'Mild',
    'mo': 'Moderate', 'moderate': 'Moderate',
    's': 'Severe', 'severe': 'Severe', 'sev': 'Severe',
    'e': 'Severe', 'extreme': 'Severe',
    '0': 'None', '1': 'Mild', '2': 'Moderate', '3': 'Severe'
}
df['addiction_level'] = df['addiction_level'].apply(lambda x: standardize_string(x, addiction_level_mapping) if pd.notna(x) else np.nan)
print("✓ Addiction level standardized")

# ============================================================================
# STEP 3: HANDLE MISSING VALUES WITH IMPUTATION
# ============================================================================
print("\n>>> STEP 3: HANDLING MISSING VALUES...")
missing_before = df.isnull().sum().sum()

# For numeric columns: Use median
numeric_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
                'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day',
                'weekend_screen_time']

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

# For categorical columns: Use mode (most frequent value)
# Preserve original addiction_level blanks so binary addicted_label stays authoritative.
categorical_cols = ['gender', 'stress_level', 'academic_work_impact']

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
        df[col].fillna(mode_val, inplace=True)

missing_after = df.isnull().sum().sum()
print(f"✓ Missing values: {missing_before} → {missing_after}")

# ============================================================================
# STEP 4: HANDLE OUTLIERS (IQR METHOD)
# ============================================================================
print("\n>>> STEP 4: HANDLING OUTLIERS...")
outliers_total = 0

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower, upper)
    outliers_total += outliers

print(f"✓ Outliers handled: {outliers_total}")

# ============================================================================
# STEP 5: APPLY REALISTIC BOUNDS
# ============================================================================
print("\n>>> STEP 5: APPLYING REALISTIC BOUNDS...")

df['age'] = df['age'].clip(13, 80)
df['daily_screen_time_hours'] = df['daily_screen_time_hours'].clip(0, 24)
df['social_media_hours'] = df['social_media_hours'].clip(0, 12)
df['gaming_hours'] = df['gaming_hours'].clip(0, 12)
df['work_study_hours'] = df['work_study_hours'].clip(0, 12)
df['sleep_hours'] = df['sleep_hours'].clip(0, 12)
df['notifications_per_day'] = df['notifications_per_day'].clip(0, 500)
df['app_opens_per_day'] = df['app_opens_per_day'].clip(0, 300)
df['weekend_screen_time'] = df['weekend_screen_time'].clip(0, 24)

print("✓ All values bounded to realistic ranges")

# ============================================================================
# STEP 6: HANDLE LABEL INCONSISTENCIES
# ============================================================================
print("\n>>> STEP 6: RECONCILING ADDICTION LABELS...")

# Preserve original addicted_label values when they exist.
if 'addicted_label' in df.columns:
    df['addicted_label'] = df['addicted_label'].astype(int)
else:
    # Create binary label from addiction_level only when no original binary label exists.
    addiction_to_binary = {
        'None': 0,
        'Mild': 1,
        'Moderate': 1,
        'Severe': 1
    }
    df['addicted_label'] = df['addiction_level'].map(addiction_to_binary).fillna(0).astype(int)

print(f"✓ Labels preserved or reconciled")
addicted_count = (df['addicted_label'] == 1).sum()
not_addicted_count = (df['addicted_label'] == 0).sum()
print(f"  • Addicted: {addicted_count:,} ({addicted_count/len(df)*100:.1f}%)")
print(f"  • Not addicted: {not_addicted_count:,} ({not_addicted_count/len(df)*100:.1f}%)")

# ============================================================================
# STEP 7: ENSURE DATA TYPE CONSISTENCY
# ============================================================================
print("\n>>> STEP 7: FIXING DATA TYPES...")

df['transaction_id'] = df['transaction_id'].astype(str)
df['user_id'] = df['user_id'].astype(str)
df['age'] = df['age'].astype(int)
df['daily_screen_time_hours'] = df['daily_screen_time_hours'].astype(float)
df['social_media_hours'] = df['social_media_hours'].astype(float)
df['gaming_hours'] = df['gaming_hours'].astype(float)
df['work_study_hours'] = df['work_study_hours'].astype(float)
df['sleep_hours'] = df['sleep_hours'].astype(float)
df['notifications_per_day'] = df['notifications_per_day'].astype(int)
df['app_opens_per_day'] = df['app_opens_per_day'].astype(int)
df['weekend_screen_time'] = df['weekend_screen_time'].astype(float)
df['addicted_label'] = df['addicted_label'].astype(int)

print("✓ Data types corrected")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 100)
print("CLEANING COMPLETE - SUMMARY")
print("=" * 100)

print(f"\n📊 DATASET TRANSFORMATION:")
print(f"  Original:  {original_shape[0]:,} rows × {original_shape[1]} columns")
print(f"  Final:     {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Records removed (duplicates): {original_shape[0] - df.shape[0]:,}")

print(f"\n✅ DATA QUALITY IMPROVEMENTS:")
print(f"  ✓ Missing values: {missing_before} → {missing_after}")
print(f"  ✓ Duplicates: {duplicates_count} removed")
print(f"  ✓ Outliers: {outliers_total} handled")
print(f"  ✓ Categorical standardization: Complete")
print(f"  ✓ Data types: All consistent")
print(f"  ✓ Value ranges: Validated")

print(f"\n📈 FEATURE STATISTICS:")
print(f"\n  Age:")
print(f"    Range: {df['age'].min()}-{df['age'].max()} years")
print(f"    Mean: {df['age'].mean():.1f}, Median: {df['age'].median():.0f}")

print(f"\n  Daily Screen Time:")
print(f"    Range: {df['daily_screen_time_hours'].min():.1f}-{df['daily_screen_time_hours'].max():.1f} hours")
print(f"    Mean: {df['daily_screen_time_hours'].mean():.2f}, Median: {df['daily_screen_time_hours'].median():.2f}")

print(f"\n  Sleep Hours:")
print(f"    Range: {df['sleep_hours'].min():.1f}-{df['sleep_hours'].max():.1f} hours")
print(f"    Mean: {df['sleep_hours'].mean():.2f}, Median: {df['sleep_hours'].median():.2f}")

print(f"\n  Notifications Per Day:")
print(f"    Range: {df['notifications_per_day'].min()}-{df['notifications_per_day'].max()}")
print(f"    Mean: {df['notifications_per_day'].mean():.0f}, Median: {df['notifications_per_day'].median():.0f}")

print(f"\n  App Opens Per Day:")
print(f"    Range: {df['app_opens_per_day'].min()}-{df['app_opens_per_day'].max()}")
print(f"    Mean: {df['app_opens_per_day'].mean():.0f}, Median: {df['app_opens_per_day'].median():.0f}")

print(f"\n📋 CATEGORICAL DISTRIBUTIONS:")
print(f"\n  Gender: {df['gender'].nunique()} categories")
for cat, count in df['gender'].value_counts().items():
    print(f"    • {cat}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n  Stress Level: {df['stress_level'].nunique()} categories")
for cat, count in df['stress_level'].value_counts().items():
    print(f"    • {cat}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n  Academic/Work Impact: {df['academic_work_impact'].nunique()} categories")
for cat, count in df['academic_work_impact'].value_counts().items():
    print(f"    • {cat}: {count:,} ({count/len(df)*100:.1f}%)")

print(f"\n  Addiction Level: {df['addiction_level'].nunique()} categories")
for cat, count in df['addiction_level'].value_counts().items():
    print(f"    • {cat}: {count:,} ({count/len(df)*100:.1f}%)")

# ============================================================================
# SAVE FINAL CLEANED DATASET
# ============================================================================
print("\n" + "=" * 100)
output_file = 'Smartphone_Dataset_Cleaned.csv'
df.to_csv(output_file, index=False)
print(f"✅ FINAL CLEANED DATASET SAVED: {output_file}")
print("=" * 100)

print("\n🎉 DATA CLEANING COMPLETE!")
print(f"   The dataset is now ready for:")
print(f"   • Exploratory Data Analysis (EDA)")
print(f"   • Feature Engineering")
print(f"   • Model Training and Evaluation")
print(f"   • Production Deployment")
