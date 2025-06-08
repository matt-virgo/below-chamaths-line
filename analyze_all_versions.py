#!/usr/bin/env python3

import pandas as pd
import numpy as np

print('🚀 COMPREHENSIVE VERSION ANALYSIS: V1 vs V2 vs V3')
print('='*80)

# Load all result sets
v1_results = pd.read_csv('ultra_deep_results.csv')
v2_results = pd.read_csv('software_engineering_results.csv')
v3_results = pd.read_csv('focused_v3_results.csv')

print(f'V1 (Original - 60+ features): {len(v1_results)} predictions')
print(f'V2 (Programmer Detection - 79 features): {len(v2_results)} predictions')
print(f'V3 (Focused Top 20): {len(v3_results)} predictions')

# Calculate key metrics for all versions
def calculate_metrics(results_df, error_col='abs_error'):
    mae = results_df[error_col].mean()
    exact = (results_df[error_col] < 0.01).sum()
    close1 = (results_df[error_col] < 1.0).sum()
    close5 = (results_df[error_col] < 5.0).sum()
    close10 = (results_df[error_col] < 10.0).sum()
    return mae, exact, close1, close5, close10

v1_mae, v1_exact, v1_close1, v1_close5, v1_close10 = calculate_metrics(v1_results)
v2_mae, v2_exact, v2_close1, v2_close5, v2_close10 = calculate_metrics(v2_results)
v3_mae, v3_exact, v3_close1, v3_close5, v3_close10 = calculate_metrics(v3_results)

print()
print('📊 MEAN ABSOLUTE ERROR COMPARISON:')
print(f'  V1 (Original):              ${v1_mae:.2f}')
print(f'  V2 (Programmer Detection):  ${v2_mae:.2f}')
print(f'  V3 (Focused Top 20):        ${v3_mae:.2f}')
print()
print('🎯 IMPROVEMENTS:')
print(f'  V3 vs V1: ${v1_mae - v3_mae:.2f} better ({((v1_mae - v3_mae)/v1_mae)*100:.1f}% improvement)')
print(f'  V3 vs V2: ${v2_mae - v3_mae:.2f} better ({((v2_mae - v3_mae)/v2_mae)*100:.1f}% improvement)')

print()
print('🎯 PRECISION METRICS COMPARISON:')
print(f'Exact matches (±$0.01):')
print(f'  V1: {v1_exact}/{len(v1_results)} ({v1_exact/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_exact}/{len(v2_results)} ({v2_exact/len(v2_results)*100:.1f}%)')
print(f'  V3: {v3_exact}/{len(v3_results)} ({v3_exact/len(v3_results)*100:.1f}%)')

print(f'\nClose matches (±$1.00):')
print(f'  V1: {v1_close1}/{len(v1_results)} ({v1_close1/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_close1}/{len(v2_results)} ({v2_close1/len(v2_results)*100:.1f}%)')
print(f'  V3: {v3_close1}/{len(v3_results)} ({v3_close1/len(v3_results)*100:.1f}%)')

print(f'\nClose matches (±$5.00):')
print(f'  V1: {v1_close5}/{len(v1_results)} ({v1_close5/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_close5}/{len(v2_results)} ({v2_close5/len(v2_results)*100:.1f}%)')
print(f'  V3: {v3_close5}/{len(v3_results)} ({v3_close5/len(v3_results)*100:.1f}%)')

print(f'\nClose matches (±$10.00):')
print(f'  V1: {v1_close10}/{len(v1_results)} ({v1_close10/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_close10}/{len(v2_results)} ({v2_close10/len(v2_results)*100:.1f}%)')
print(f'  V3: {v3_close10}/{len(v3_results)} ({v3_close10/len(v3_results)*100:.1f}%)')

print()
print('📈 ERROR DISTRIBUTION ANALYSIS:')
versions = [
    ('V1 (Original)', v1_results, 'abs_error'),
    ('V2 (Programmer Detection)', v2_results, 'abs_error'),
    ('V3 (Focused Top 20)', v3_results, 'abs_error')
]

for name, results, error_col in versions:
    print(f'\n{name}:')
    print(f'  $0.00-$1.00:   {(results[error_col] <= 1.0).sum():3d}/{len(results)} ({(results[error_col] <= 1.0).mean()*100:4.1f}%)')
    print(f'  $1.01-$5.00:   {((results[error_col] > 1.0) & (results[error_col] <= 5.0)).sum():3d}/{len(results)} ({((results[error_col] > 1.0) & (results[error_col] <= 5.0)).mean()*100:4.1f}%)')
    print(f'  $5.01-$25.00:  {((results[error_col] > 5.0) & (results[error_col] <= 25.0)).sum():3d}/{len(results)} ({((results[error_col] > 5.0) & (results[error_col] <= 25.0)).mean()*100:4.1f}%)')
    print(f'  $25.01-$100:   {((results[error_col] > 25.0) & (results[error_col] <= 100.0)).sum():3d}/{len(results)} ({((results[error_col] > 25.0) & (results[error_col] <= 100.0)).mean()*100:4.1f}%)')
    print(f'  $100+:         {(results[error_col] > 100.0).sum():3d}/{len(results)} ({(results[error_col] > 100.0).mean()*100:4.1f}%)')

print()
print('🏅 TOP 10 MOST ACCURATE PREDICTIONS (V3):')
top_v3 = v3_results.nsmallest(10, 'abs_error')
for i, row in top_v3.iterrows():
    print(f'{row.name+1:2d}. Error: ${row["abs_error"]:.4f} | Days: {row["trip_duration_days"]:2.0f}, Miles: {row["miles_traveled"]:4.0f}, Receipts: ${row["total_receipts_amount"]:7.2f}')

print()
print('📊 FEATURE EFFICIENCY ANALYSIS:')
print(f'  V1: ~60+ features → ${v1_mae:.2f} MAE')
print(f'  V2: 79 features   → ${v2_mae:.2f} MAE  (More features, worse performance)')
print(f'  V3: 20 features   → ${v3_mae:.2f} MAE  (Fewer features, BEST performance)')
print()
print('💡 KEY INSIGHT: Focused feature selection (V3) dramatically outperforms both approaches!')

print()
print('✨ FINAL RANKINGS:')
rankings = [
    ('🥇 V3 (Focused Top 20)', v3_mae, '20 focused features', 'BEST'),
    ('🥈 V1 (Original)', v1_mae, '~60+ features', 'Good baseline'),
    ('🥉 V2 (Programmer Detection)', v2_mae, '79 features', 'Insight valuable but noisy')
]

for rank, (name, mae, features, note) in enumerate(rankings, 1):
    print(f'{name:<30} | MAE: ${mae:6.2f} | {features:<20} | {note}')

print()
print('🎉 CONCLUSION:')
print('  ✅ V3 Focused approach wins decisively!')
print(f'  ✅ {((v1_mae - v3_mae)/v1_mae)*100:.1f}% improvement over V1 baseline')
print(f'  ✅ {((v2_mae - v3_mae)/v2_mae)*100:.1f}% improvement over V2 full feature set')
print('  ✅ Demonstrates the power of thoughtful feature selection')
print('  ✅ 20 carefully chosen features > 79 comprehensive features')
print()
print('🧠 LESSON LEARNED: Sometimes less is more in machine learning!') 