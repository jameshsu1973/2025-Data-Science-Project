import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Load first 1000 rows
df = pd.read_csv('dataset/train_ver2.csv', nrows=1000)

print('='*70)
print('Dataset Info')
print('='*70)
print(f'Shape: {df.shape}')

print('\n' + '='*70)
print('Missing Values (NaN)')
print('='*70)
missing = df.isnull().sum()
if missing[missing > 0].any():
    print(missing[missing > 0])
else:
    print('No missing values')

print('\n' + '='*70)
print('Empty Strings')
print('='*70)
has_empty = False
for col in df.select_dtypes(include=['object']).columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        print(f'{col:30s}: {empty_count:4d} ({empty_count/len(df)*100:.1f}%)')
        has_empty = True

if not has_empty:
    print('No empty strings found')

print('\n' + '='*70)
print('Sample of categorical columns')
print('='*70)
categorical_cols = df.select_dtypes(include=['object']).columns[:5]
for col in categorical_cols:
    print(f'\n{col}:')
    print(df[col].value_counts().head(5))
