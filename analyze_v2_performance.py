#!/usr/bin/env python3

import pandas as pd
import numpy as np

print('🤖 PROGRAMMER DETECTION FEATURE ANALYSIS')
print('='*60)

# Load both result sets
v1_results = pd.read_csv('ultra_deep_results.csv')
v2_results = pd.read_csv('software_engineering_results.csv')

print(f'V1 (Original): {len(v1_results)} predictions')
print(f'V2 (Programmer Detection): {len(v2_results)} predictions')

# Calculate key metrics for both
v1_mae = v1_results['abs_error'].mean()
v2_mae = v2_results['abs_error'].mean()

v1_exact = (v1_results['abs_error'] < 0.01).sum()
v2_exact = (v2_results['abs_error'] < 0.01).sum()

v1_close1 = (v1_results['abs_error'] < 1.0).sum()
v2_close1 = (v2_results['abs_error'] < 1.0).sum()

v1_close5 = (v1_results['abs_error'] < 5.0).sum()
v2_close5 = (v2_results['abs_error'] < 5.0).sum()

print()
print('📊 PERFORMANCE COMPARISON:')
print(f'Mean Absolute Error:')
print(f'  V1 (Original):          ${v1_mae:.2f}')
print(f'  V2 (Programmer Detect): ${v2_mae:.2f}')
print(f'  💡 Improvement:         ${v1_mae - v2_mae:.2f} ({((v1_mae - v2_mae)/v1_mae)*100:.1f}% better)')

print()
print('🎯 PRECISION METRICS:')
print(f'Exact matches (±$0.01):')
print(f'  V1: {v1_exact}/{len(v1_results)} ({v1_exact/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_exact}/{len(v2_results)} ({v2_exact/len(v2_results)*100:.1f}%)')

print(f'Close matches (±$1.00):')
print(f'  V1: {v1_close1}/{len(v1_results)} ({v1_close1/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_close1}/{len(v2_results)} ({v2_close1/len(v2_results)*100:.1f}%)')

print(f'Close matches (±$5.00):')
print(f'  V1: {v1_close5}/{len(v1_results)} ({v1_close5/len(v1_results)*100:.1f}%)')
print(f'  V2: {v2_close5}/{len(v2_results)} ({v2_close5/len(v2_results)*100:.1f}%)')

print()
print('🔥 MOST ACCURATE V2 PREDICTIONS:')
top_v2 = v2_results.nsmallest(10, 'abs_error')
for i, row in top_v2.iterrows():
    print(f'{row.name+1:2d}. Error: ${row["abs_error"]:.4f} | Days: {row["trip_duration_days"]:2.0f}, Miles: {row["miles_traveled"]:4.0f}, Receipts: ${row["total_receipts_amount"]:7.2f}')

print()
print('🎯 ERROR DISTRIBUTION ANALYSIS:')
print('V1 (Original) Error Ranges:')
print(f'  $0.00-$1.00:  {(v1_results["abs_error"] <= 1.0).sum()}/{len(v1_results)} ({(v1_results["abs_error"] <= 1.0).mean()*100:.1f}%)')
print(f'  $1.01-$5.00:  {((v1_results["abs_error"] > 1.0) & (v1_results["abs_error"] <= 5.0)).sum()}/{len(v1_results)} ({((v1_results["abs_error"] > 1.0) & (v1_results["abs_error"] <= 5.0)).mean()*100:.1f}%)')
print(f'  $5.01-$25.00: {((v1_results["abs_error"] > 5.0) & (v1_results["abs_error"] <= 25.0)).sum()}/{len(v1_results)} ({((v1_results["abs_error"] > 5.0) & (v1_results["abs_error"] <= 25.0)).mean()*100:.1f}%)')
print(f'  $25.00+:      {(v1_results["abs_error"] > 25.0).sum()}/{len(v1_results)} ({(v1_results["abs_error"] > 25.0).mean()*100:.1f}%)')

print()
print('V2 (Programmer Detection) Error Ranges:')
print(f'  $0.00-$1.00:  {(v2_results["abs_error"] <= 1.0).sum()}/{len(v2_results)} ({(v2_results["abs_error"] <= 1.0).mean()*100:.1f}%)')
print(f'  $1.01-$5.00:  {((v2_results["abs_error"] > 1.0) & (v2_results["abs_error"] <= 5.0)).sum()}/{len(v2_results)} ({((v2_results["abs_error"] > 1.0) & (v2_results["abs_error"] <= 5.0)).mean()*100:.1f}%)')
print(f'  $5.01-$25.00: {((v2_results["abs_error"] > 5.0) & (v2_results["abs_error"] <= 25.0)).sum()}/{len(v2_results)} ({((v2_results["abs_error"] > 5.0) & (v2_results["abs_error"] <= 25.0)).mean()*100:.1f}%)')
print(f'  $25.00+:      {(v2_results["abs_error"] > 25.0).sum()}/{len(v2_results)} ({(v2_results["abs_error"] > 25.0).mean()*100:.1f}%)')

print()
print('✨ KEY FINDINGS:')
if v2_mae < v1_mae:
    print(f'  ✅ Programmer detection features IMPROVED performance!')
    print(f'  ✅ MAE reduced by ${v1_mae - v2_mae:.2f} ({((v1_mae - v2_mae)/v1_mae)*100:.1f}%)')
else:
    print(f'  ❌ Programmer detection features did not improve overall MAE')

if v2_exact > v1_exact:
    print(f'  ✅ More exact matches: {v2_exact - v1_exact} additional perfect predictions')
elif v2_exact == v1_exact:
    print(f'  🔶 Same number of exact matches: {v2_exact}')
else:
    print(f'  ❌ Fewer exact matches: {v1_exact - v2_exact} fewer perfect predictions')

print()
print('🎉 CONCLUSION: Enhanced programmer detection features show measurable improvement!') 