#!/usr/bin/env python3

"""
TabPFN Business Rules Champion - Private Results Generation
Train once on public cases, then efficiently batch process private cases

This script:
1. Loads public cases for training (like we did in batch evaluation)
2. Trains TabPFN once with business rules features
3. Loads private cases and processes them in batch
4. Outputs results to private_results.txt in the required format
"""

import json
import math
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_public_cases():
    """Load all public cases for training"""
    print("📊 Loading public cases for training...")
    
    with open('public_cases.json', 'r') as f:
        public_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        for case in public_data
    ])
    
    print(f"   ✅ Loaded {len(df)} public cases for training")
    return df

def load_private_cases():
    """Load private cases for prediction"""
    print("🔒 Loading private cases for prediction...")
    
    with open('private_cases.json', 'r') as f:
        private_data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'trip_duration_days': case['trip_duration_days'],
            'miles_traveled': case['miles_traveled'],
            'total_receipts_amount': case['total_receipts_amount']
        }
        for case in private_data
    ])
    
    print(f"   ✅ Loaded {len(df)} private cases for prediction")
    return df

def engineer_business_features(df_input):
    """Apply our winning business rules feature engineering (31 features)"""
    df = df_input.copy()

    # Ensure trip_duration_days is at least 1 to avoid division by zero
    df['trip_duration_days_safe'] = df['trip_duration_days'].apply(lambda x: x if x > 0 else 1)

    # Base engineered features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days_safe']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days_safe']
    
    df['receipt_cents_val'] = df['total_receipts_amount'].apply(
        lambda x: round((x - math.floor(x)) * 100) if isinstance(x, (int, float)) and not math.isnan(x) else 0
    )
    df['is_receipt_49_or_99_cents'] = df['receipt_cents_val'].apply(lambda x: 1 if x == 49 or x == 99 else 0).astype(int)
    
    # Trip length categories
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_short_trip'] = (df['trip_duration_days'] < 4).astype(int)
    df['is_medium_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['is_long_trip'] = ((df['trip_duration_days'] > 6) & (df['trip_duration_days'] < 8)).astype(int)
    df['is_very_long_trip'] = (df['trip_duration_days'] >= 8).astype(int)

    # Polynomial features
    df['trip_duration_sq'] = df['trip_duration_days']**2
    df['miles_traveled_sq'] = df['miles_traveled']**2
    df['total_receipts_amount_sq'] = df['total_receipts_amount']**2
    df['miles_per_day_sq'] = df['miles_per_day']**2
    df['receipts_per_day_sq'] = df['receipts_per_day']**2

    # Mileage-based features
    df['miles_first_100'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_after_100'] = df['miles_traveled'].apply(lambda x: max(0, x - 100))
    df['is_high_mileage_trip'] = (df['miles_traveled'] > 500).astype(int)

    # Receipt-based features
    df['is_very_low_receipts_multiday'] = ((df['total_receipts_amount'] < 50) & (df['trip_duration_days'] > 1)).astype(int)
    df['is_moderate_receipts'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['is_high_receipts'] = ((df['total_receipts_amount'] > 800) & (df['total_receipts_amount'] <= 1200)).astype(int)
    df['is_very_high_receipts'] = (df['total_receipts_amount'] > 1200).astype(int)

    # Kevin's insights
    df['is_optimal_miles_per_day_kevin'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    
    def optimal_daily_spending(row):
        if row['is_short_trip']:
            return 1 if row['receipts_per_day'] < 75 else 0
        elif row['is_medium_trip']:
            return 1 if row['receipts_per_day'] < 120 else 0
        elif row['is_long_trip'] or row['is_very_long_trip']: 
            return 1 if row['receipts_per_day'] < 90 else 0
        return 0 
    df['is_optimal_daily_spending_kevin'] = df.apply(optimal_daily_spending, axis=1).astype(int)

    # Interaction features
    df['duration_x_miles_per_day'] = df['trip_duration_days'] * df['miles_per_day']
    df['receipts_per_day_x_duration'] = df['receipts_per_day'] * df['trip_duration_days']
    
    df['interaction_kevin_sweet_spot'] = (df['is_5_day_trip'] & \
                                         (df['miles_per_day'] >= 180) & \
                                         (df['receipts_per_day'] < 100)).astype(int)
    
    df['interaction_kevin_vacation_penalty'] = (df['is_very_long_trip'] & \
                                               (df['receipts_per_day'] > 90)).astype(int)

    df['interaction_efficiency_metric'] = df['miles_traveled'] / (df['trip_duration_days_safe']**0.5 + 1e-6) 
    df['interaction_spending_mileage_ratio'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1e-6)

    # Select final features (31 total)
    business_features = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_per_day', 'receipts_per_day', 
        'is_receipt_49_or_99_cents',
        'is_5_day_trip', 'is_short_trip', 'is_medium_trip', 'is_long_trip', 'is_very_long_trip',
        'trip_duration_sq', 'miles_traveled_sq', 'total_receipts_amount_sq', 'miles_per_day_sq', 'receipts_per_day_sq',
        'miles_first_100', 'miles_after_100', 'is_high_mileage_trip',
        'is_very_low_receipts_multiday', 'is_moderate_receipts', 'is_high_receipts', 'is_very_high_receipts',
        'is_optimal_miles_per_day_kevin', 'is_optimal_daily_spending_kevin',
        'duration_x_miles_per_day', 'receipts_per_day_x_duration',
        'interaction_kevin_sweet_spot', 'interaction_kevin_vacation_penalty',
        'interaction_efficiency_metric', 'interaction_spending_mileage_ratio'
    ]
    
    return df[business_features]

def analyze_private_patterns(X_features):
    """Analyze business patterns in the private dataset"""
    print("\n🔍 PRIVATE DATASET BUSINESS PATTERN ANALYSIS:")
    
    # Kevin's patterns
    sweet_spots = (X_features['interaction_kevin_sweet_spot'] == 1).sum()
    vacation_penalties = (X_features['interaction_kevin_vacation_penalty'] == 1).sum()
    optimal_miles = (X_features['is_optimal_miles_per_day_kevin'] == 1).sum()
    optimal_spending = (X_features['is_optimal_daily_spending_kevin'] == 1).sum()
    lucky_cents = (X_features['is_receipt_49_or_99_cents'] == 1).sum()
    
    total = len(X_features)
    
    print(f"   🎯 Kevin's Sweet Spot trips: {sweet_spots} ({sweet_spots/total*100:.1f}%)")
    print(f"   ⚠️  Vacation Penalty trips: {vacation_penalties} ({vacation_penalties/total*100:.1f}%)")
    print(f"   🛣️  Optimal mileage (180-220/day): {optimal_miles} ({optimal_miles/total*100:.1f}%)")
    print(f"   💰 Optimal spending patterns: {optimal_spending} ({optimal_spending/total*100:.1f}%)")
    print(f"   🍀 Lucky cents (49/99): {lucky_cents} ({lucky_cents/total*100:.1f}%)")
    
    # Trip distribution
    trip_dist = {
        'Short (<4 days)': (X_features['is_short_trip'] == 1).sum(),
        'Medium (4-6 days)': (X_features['is_medium_trip'] == 1).sum(), 
        'Long (7 days)': (X_features['is_long_trip'] == 1).sum(),
        'Very Long (8+ days)': (X_features['is_very_long_trip'] == 1).sum()
    }
    
    print(f"\n   📅 Trip Duration Distribution:")
    for trip_type, count in trip_dist.items():
        print(f"      {trip_type}: {count} ({count/total*100:.1f}%)")

def main():
    print("🏆 TabPFN Business Rules Champion - Private Results Generation")
    print("="*80)
    print("Training once on public cases, then batch-processing private cases")
    print("Expected: World Record quality predictions for submission!")
    print()
    
    try:
        from tabpfn import TabPFNRegressor
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
        return
    
    # Load training data (public cases)
    start_time = time.time()
    public_df = load_public_cases()
    
    # Engineer training features
    print(f"\n{'='*80}")
    print(f"🏢 TRAINING ON PUBLIC CASES")
    print(f"{'='*80}")
    
    X_train = engineer_business_features(public_df)
    y_train = public_df['reimbursement'].values
    
    print(f"\n✨ CHAMPION TRAINING SET:")
    print(f"   🏆 Features: {X_train.shape[1]}")
    print(f"   📊 Training samples: {len(X_train)}")
    
    # Train TabPFN
    print(f"\n🚀 Training TabPFN Champion...")
    print(f"   📊 Training on {len(X_train)} public cases")
    print(f"   🏢 Using {X_train.shape[1]} business-engineered features")
    
    tabpfn = TabPFNRegressor(device='cpu')
    
    train_start = time.time()
    X_train_np = X_train.values.astype(np.float32)
    y_train_np = y_train.astype(np.float32)
    
    tabpfn.fit(X_train_np, y_train_np)
    train_time = time.time() - train_start
    
    print(f"   ✅ Training completed in {train_time:.2f} seconds")
    
    # Load and process private cases
    print(f"\n{'='*80}")
    print(f"🔒 PROCESSING PRIVATE CASES")
    print(f"{'='*80}")
    
    private_df = load_private_cases()
    X_private = engineer_business_features(private_df)
    
    print(f"\n✨ PRIVATE DATASET PREPARED:")
    print(f"   🔒 Cases to predict: {len(X_private)}")
    print(f"   🏢 Features: {X_private.shape[1]} (same as training)")
    
    # Analyze private patterns
    analyze_private_patterns(X_private)
    
    # Batch prediction
    print(f"\n{'='*80}")
    print(f"🔮 BATCH PREDICTION")
    print(f"{'='*80}")
    
    print(f"🚀 Generating predictions for all {len(X_private)} private cases...")
    pred_start = time.time()
    
    X_private_np = X_private.values.astype(np.float32)
    y_pred = tabpfn.predict(X_private_np)
    
    pred_time = time.time() - pred_start
    total_time = time.time() - start_time
    
    print(f"   ✅ Batch prediction completed in {pred_time:.2f} seconds")
    print(f"   ⚡ Total runtime: {total_time:.2f} seconds")
    print(f"   🎯 Speed: {len(X_private)/pred_time:.1f} predictions/second")
    
    # Save results to private_results.txt
    print(f"\n{'='*80}")
    print(f"💾 SAVING RESULTS")
    print(f"{'='*80}")
    
    with open('private_results.txt', 'w') as f:
        for prediction in y_pred:
            f.write(f"{prediction:.2f}\n")
    
    print(f"✅ Results saved to private_results.txt")
    print(f"📊 Format: One prediction per line ({len(y_pred)} lines total)")
    print(f"🎯 Each line corresponds to same-numbered case in private_cases.json")
    
    # Analysis summary
    min_pred = y_pred.min()
    max_pred = y_pred.max()
    mean_pred = y_pred.mean()
    std_pred = y_pred.std()
    
    print(f"\n{'='*80}")
    print(f"📈 PREDICTION SUMMARY")
    print(f"{'='*80}")
    
    print(f"📊 Prediction Statistics:")
    print(f"   Minimum: ${min_pred:.2f}")
    print(f"   Maximum: ${max_pred:.2f}")
    print(f"   Mean: ${mean_pred:.2f}")
    print(f"   Std Dev: ${std_pred:.2f}")
    
    # Sample predictions
    print(f"\n📋 Sample Predictions (first 5 cases):")
    for i in range(min(5, len(y_pred))):
        row = private_df.iloc[i]
        print(f"   Case {i+1}: {row['trip_duration_days']} days, {row['miles_traveled']} miles, ${row['total_receipts_amount']:.2f} → ${y_pred[i]:.2f}")
    
    print(f"\n🎉 SUBMISSION READY!")
    print(f"   📄 File: private_results.txt")
    print(f"   📊 Lines: {len(y_pred)}")
    print(f"   🏆 Model: TabPFN Business Rules Champion")
    print(f"   ⚡ Processing: {len(X_private)/pred_time:.1f} predictions/second")
    print(f"   🎯 Expected Quality: World Record level (based on $43.94 MAE public performance)")

if __name__ == "__main__":
    main() 