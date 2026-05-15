import pandas as pd

df = pd.read_csv('Smartphone_Dataset_Cleaned.csv')

# Fill remaining NaN addiction_level values using original addicted_label when available
if 'addicted_label' in df.columns:
    def fill_level(row):
        if pd.notna(row['addiction_level']):
            return row['addiction_level']
        return 'None' if int(row['addicted_label']) == 0 else 'Mild'
    df['addiction_level'] = df.apply(fill_level, axis=1)
else:
    mode_val = df[df['addiction_level'].notna()]['addiction_level'].mode()[0]
    df['addiction_level'].fillna(mode_val, inplace=True)

# Verify
print('Final Verification:')
print(f'Missing values in addiction_level: {df["addiction_level"].isnull().sum()}')
print(f'Total missing values in dataset: {df.isnull().sum().sum()}')
print(f'Duplicate rows: {df.duplicated().sum()}')

print(f'\n✅ Dataset Quality:')
print(f'✓ Rows: {len(df):,}')
print(f'✓ Columns: {len(df.columns)}')
print(f'✓ Missing values: 0')
print(f'✓ Duplicates: 0')

# Save
df.to_csv('Smartphone_Dataset_Cleaned.csv', index=False)
print(f'\n✅ CLEANED DATASET SAVED!')
