#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🔍 Specific Outlier Analysis")
    print("="*70)
    print("Analyzing the characteristics of detected outliers")
    print()
    
    # Load the comprehensive outlier analysis
    print("Loading outlier analysis results...")
    df = pd.read_csv('comprehensive_outlier_analysis.csv')
    
    print(f"Dataset overview:")
    print(f"   📊 Total samples: {len(df)}")
    print(f"   📊 Train samples: {len(df[df['dataset'] == 'train'])}")
    print(f"   📊 Test samples: {len(df[df['dataset'] == 'test'])}")
    
    # Analyze isolation forest outliers (the main statistical outliers detected)
    iso_outliers = df[df['isolation_forest_outliers'] == True]
    normal_samples = df[df['isolation_forest_outliers'] == False]
    
    print(f"\n{'='*70}")
    print(f"🚨 ISOLATION FOREST OUTLIERS ANALYSIS")
    print(f"{'='*70}")
    
    print(f"Isolation Forest outliers: {len(iso_outliers)}/{len(df)} ({len(iso_outliers)/len(df)*100:.1f}%)")
    
    # Break down by dataset
    iso_train_outliers = iso_outliers[iso_outliers['dataset'] == 'train']
    iso_test_outliers = iso_outliers[iso_outliers['dataset'] == 'test']
    
    print(f"   Train outliers: {len(iso_train_outliers)}")
    print(f"   Test outliers: {len(iso_test_outliers)}")
    
    # Show characteristics of outliers vs normal
    print(f"\n📊 OUTLIER CHARACTERISTICS:")
    features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'reimbursement']
    
    for feature in features:
        outlier_stats = iso_outliers[feature].describe()
        normal_stats = normal_samples[feature].describe()
        
        print(f"\n{feature.upper()}:")
        print(f"   Outliers - Mean: {outlier_stats['mean']:.2f}, Median: {outlier_stats['50%']:.2f}, Std: {outlier_stats['std']:.2f}")
        print(f"   Normal   - Mean: {normal_stats['mean']:.2f}, Median: {normal_stats['50%']:.2f}, Std: {normal_stats['std']:.2f}")
        print(f"   Range outliers: {outlier_stats['min']:.2f} - {outlier_stats['max']:.2f}")
        print(f"   Range normal:   {normal_stats['min']:.2f} - {normal_stats['max']:.2f}")
    
    # Show most extreme outliers
    print(f"\n🚨 TOP 20 ISOLATION FOREST OUTLIERS (Most Extreme):")
    iso_sorted = iso_outliers.sort_values('isolation_forest_score')
    
    for i, (idx, row) in enumerate(iso_sorted.head(20).iterrows()):
        dataset_type = row['dataset']
        original_idx = row['index']
        score = row['isolation_forest_score']
        print(f"   {i+1:2d}. {dataset_type.upper()}[{original_idx:3d}] | "
              f"Days: {row['trip_duration_days']:2d} | "
              f"Miles: {row['miles_traveled']:6.1f} | "
              f"Receipts: ${row['total_receipts_amount']:7.2f} | "
              f"Reimbursement: ${row['reimbursement']:8.2f} | "
              f"Score: {score:.4f}")
    
    # Analyze high prediction error outliers
    if 'high_error_outlier' in df.columns:
        high_error_outliers = df[df['high_error_outlier'] == True]
        print(f"\n{'='*70}")
        print(f"📈 HIGH PREDICTION ERROR OUTLIERS")
        print(f"{'='*70}")
        
        print(f"High error outliers: {len(high_error_outliers)}")
        
        if len(high_error_outliers) > 0:
            print(f"\n🎯 TOP HIGH ERROR SAMPLES:")
            error_sorted = high_error_outliers.sort_values('prediction_error', ascending=False)
            
            for i, (idx, row) in enumerate(error_sorted.head(15).iterrows()):
                dataset_type = row['dataset']
                original_idx = row['index']
                error = row['prediction_error']
                is_iso_outlier = row['isolation_forest_outliers']
                outlier_indicator = " 🚨" if is_iso_outlier else ""
                
                print(f"   {i+1:2d}. {dataset_type.upper()}[{original_idx:3d}] | "
                      f"Days: {row['trip_duration_days']:2d} | "
                      f"Miles: {row['miles_traveled']:6.1f} | "
                      f"Receipts: ${row['total_receipts_amount']:7.2f} | "
                      f"Reimbursement: ${row['reimbursement']:8.2f} | "
                      f"Error: ${error:.2f}{outlier_indicator}")
    
    # Look for patterns in outliers
    print(f"\n{'='*70}")
    print(f"🔍 OUTLIER PATTERN ANALYSIS")
    print(f"{'='*70}")
    
    # Analyze extreme values in each dimension
    print("Extreme value analysis:")
    
    # Trip duration patterns
    long_trips = iso_outliers[iso_outliers['trip_duration_days'] >= 13]
    short_trips = iso_outliers[iso_outliers['trip_duration_days'] <= 2]
    print(f"   Very long trips (≥13 days): {len(long_trips)}")
    print(f"   Very short trips (≤2 days): {len(short_trips)}")
    
    # Miles patterns
    low_miles = iso_outliers[iso_outliers['miles_traveled'] <= 100]
    high_miles = iso_outliers[iso_outliers['miles_traveled'] >= 1000]
    print(f"   Low miles (≤100): {len(low_miles)}")
    print(f"   High miles (≥1000): {len(high_miles)}")
    
    # Receipts patterns
    low_receipts = iso_outliers[iso_outliers['total_receipts_amount'] <= 100]
    high_receipts = iso_outliers[iso_outliers['total_receipts_amount'] >= 2000]
    print(f"   Low receipts (≤$100): {len(low_receipts)}")
    print(f"   High receipts (≥$2000): {len(high_receipts)}")
    
    # Reimbursement patterns
    low_reimbursement = iso_outliers[iso_outliers['reimbursement'] <= 500]
    high_reimbursement = iso_outliers[iso_outliers['reimbursement'] >= 1800]
    print(f"   Low reimbursement (≤$500): {len(low_reimbursement)}")
    print(f"   High reimbursement (≥$1800): {len(high_reimbursement)}")
    
    # Look for unusual ratios
    print(f"\nUnusual ratio analysis:")
    iso_outliers_copy = iso_outliers.copy()
    iso_outliers_copy['miles_per_day'] = iso_outliers_copy['miles_traveled'] / iso_outliers_copy['trip_duration_days']
    iso_outliers_copy['receipts_per_day'] = iso_outliers_copy['total_receipts_amount'] / iso_outliers_copy['trip_duration_days']
    iso_outliers_copy['receipts_per_mile'] = iso_outliers_copy['total_receipts_amount'] / (iso_outliers_copy['miles_traveled'] + 1)
    
    low_mpd = iso_outliers_copy[iso_outliers_copy['miles_per_day'] <= 20]
    high_mpd = iso_outliers_copy[iso_outliers_copy['miles_per_day'] >= 200]
    print(f"   Very low miles/day (≤20): {len(low_mpd)}")
    print(f"   Very high miles/day (≥200): {len(high_mpd)}")
    
    low_rpd = iso_outliers_copy[iso_outliers_copy['receipts_per_day'] <= 50]
    high_rpd = iso_outliers_copy[iso_outliers_copy['receipts_per_day'] >= 200]
    print(f"   Very low receipts/day (≤$50): {len(low_rpd)}")
    print(f"   Very high receipts/day (≥$200): {len(high_rpd)}")
    
    # Identify specific problematic cases
    print(f"\n🎯 SPECIFIC PROBLEMATIC CASES:")
    
    # Case 1: Very low activity
    low_activity = iso_outliers[(iso_outliers['miles_traveled'] <= 100) & 
                               (iso_outliers['total_receipts_amount'] <= 100)]
    if len(low_activity) > 0:
        print(f"   Low activity trips (low miles + low receipts): {len(low_activity)}")
        for i, (idx, row) in enumerate(low_activity.head(5).iterrows()):
            print(f"      {i+1}. {row['dataset'].upper()}[{row['index']}]: {row['trip_duration_days']}d, {row['miles_traveled']:.0f}mi, ${row['total_receipts_amount']:.2f}, reimb: ${row['reimbursement']:.2f}")
    
    # Case 2: High receipts, low miles
    high_receipts_low_miles = iso_outliers[(iso_outliers['miles_traveled'] <= 200) & 
                                          (iso_outliers['total_receipts_amount'] >= 1500)]
    if len(high_receipts_low_miles) > 0:
        print(f"   High receipts but low miles: {len(high_receipts_low_miles)}")
        for i, (idx, row) in enumerate(high_receipts_low_miles.head(5).iterrows()):
            print(f"      {i+1}. {row['dataset'].upper()}[{row['index']}]: {row['trip_duration_days']}d, {row['miles_traveled']:.0f}mi, ${row['total_receipts_amount']:.2f}, reimb: ${row['reimbursement']:.2f}")
    
    # Case 3: High miles, low receipts
    high_miles_low_receipts = iso_outliers[(iso_outliers['miles_traveled'] >= 800) & 
                                          (iso_outliers['total_receipts_amount'] <= 500)]
    if len(high_miles_low_receipts) > 0:
        print(f"   High miles but low receipts: {len(high_miles_low_receipts)}")
        for i, (idx, row) in enumerate(high_miles_low_receipts.head(5).iterrows()):
            print(f"      {i+1}. {row['dataset'].upper()}[{row['index']}]: {row['trip_duration_days']}d, {row['miles_traveled']:.0f}mi, ${row['total_receipts_amount']:.2f}, reimb: ${row['reimbursement']:.2f}")
    
    # Case 4: Very short/long trips with unusual patterns
    extreme_duration = iso_outliers[(iso_outliers['trip_duration_days'] <= 1) | 
                                   (iso_outliers['trip_duration_days'] >= 13)]
    if len(extreme_duration) > 0:
        print(f"   Extreme duration trips: {len(extreme_duration)}")
        for i, (idx, row) in enumerate(extreme_duration.head(5).iterrows()):
            print(f"      {i+1}. {row['dataset'].upper()}[{row['index']}]: {row['trip_duration_days']}d, {row['miles_traveled']:.0f}mi, ${row['total_receipts_amount']:.2f}, reimb: ${row['reimbursement']:.2f}")
    
    # Summary insights
    print(f"\n{'='*70}")
    print(f"🧠 OUTLIER INSIGHTS SUMMARY")
    print(f"{'='*70}")
    
    print("Key outlier characteristics identified:")
    print("   1. 🔴 Ultra low activity: Very few miles and very low receipts")
    print("   2. 🟠 Unbalanced effort: High receipts with low miles or vice versa")
    print("   3. 🟡 Extreme durations: Very short (1 day) or very long (14 days) trips")
    print("   4. 🟢 Unusual ratios: Miles/day or receipts/day outside normal ranges")
    
    print(f"\nData quality concerns:")
    print("   • Some samples may have data entry errors")
    print("   • Edge cases might represent unusual but valid business scenarios")
    print("   • Low activity trips might be incomplete records")
    print("   • High receipts with low miles could indicate conference/meeting trips")
    
    print(f"\nModel impact:")
    print("   • These outliers make it harder for models to learn consistent patterns")
    print("   • They contribute to higher prediction errors")
    print("   • They represent the hardest cases to predict accurately")
    
    print(f"\n🚀 RECOMMENDATIONS:")
    print("   1. 🔍 Manual review of the most extreme outliers for data quality")
    print("   2. 🧹 Consider removing clear data errors from training")
    print("   3. 🏷️  Create separate models for different trip types (business vs travel)")
    print("   4. ⚖️  Use outlier-robust loss functions (Huber loss, etc.)")
    print("   5. 📊 Weight training samples inversely to their outlier scores")
    print("   6. 🎯 Use ensemble methods that are naturally robust to outliers")

if __name__ == "__main__":
    main() 