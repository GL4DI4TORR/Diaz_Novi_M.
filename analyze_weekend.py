import pandas as pd

# Load dataset
df = pd.read_csv('Smartphone_Dataset.csv')

# Analyze weekend screen time
print('Weekend Screen Time Analysis:')
print('=' * 40)
print(df['weekend_screen_time'].describe())
print(f'\nAverage weekend screen time: {df["weekend_screen_time"].mean():.2f} hours')
print(f'Median weekend screen time: {df["weekend_screen_time"].median():.2f} hours')

# Also analyze daily screen time for comparison
print('\nDaily Screen Time Analysis:')
print('=' * 40)
print(df['daily_screen_time_hours'].describe())
print(f'\nAverage daily screen time: {df["daily_screen_time_hours"].mean():.2f} hours')

# Calculate ratio
print('\nWeekend vs Daily Ratio:')
print('=' * 40)
ratio = df['weekend_screen_time'].mean() / df['daily_screen_time_hours'].mean()
print(f'Weekend time is {ratio:.2f}x daily time')
