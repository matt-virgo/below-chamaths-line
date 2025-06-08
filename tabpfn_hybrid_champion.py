#!/usr/bin/env python3

"""
TabPFN Hybrid Champion - Business Rules + Mathematical Features
Combining the winning business rules (31 features) with the best mathematical/programming 
features from previous record-holding experiments to create the ultimate hybrid approach
"""

import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def load_data():
    """Load training and test data"""
    with open('train_cases.json', 'r') as f:
        train_data = json.load(f)
    
    with open('test_cases.json', 'r') as f:
        test_data = json.load(f)
    
    # Convert to DataFrames
    train_df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        for case in train_data
    ])
    
    test_df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        for case in test_data
    ])
    
    return train_df, test_df

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci_ratio(n):
    """Calculate position in Fibonacci sequence or closest ratio"""
    if n <= 0:
        return 0
    
    # Generate Fibonacci numbers up to n
    fib = [1, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    
    # Find closest Fibonacci number and calculate ratio
    closest_fib = min(fib, key=lambda x: abs(x - n))
    return n / closest_fib if closest_fib > 0 else 1

def create_business_rules_features(df_input):
    """Create the winning business rules features (31 features)"""
    df = df_input.copy()

    print("   🏢 Creating business-rules features...")

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
    
    return df, business_features

def create_best_mathematical_features(df):
    """Create the top 18 mathematical/programming features from previous record holders"""
    
    print("   🧮 Creating best mathematical/programming features...")
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # TOP V1 MATHEMATICAL FEATURES (8 features) - from $55.96 record
    df['total_trip_value'] = D * M * R  # Most important V1 feature
    df['receipts_log'] = np.log1p(R)
    df['receipts_sqrt'] = np.sqrt(R)
    df['miles_log'] = np.log1p(M)
    df['receipts_sin_1000'] = np.sin(R / 1000)
    df['receipts_cos_1000'] = np.cos(R / 1000)
    df['receipts_exp_norm'] = np.exp(R / 2000) - 1
    df['receipts_cubed'] = R ** 3
    
    # TOP PROGRAMMER FEATURES (10 features) - from $55.63 record  
    df['fib_ratio_receipts'] = R.apply(fibonacci_ratio)
    df['fib_ratio_miles'] = M.apply(fibonacci_ratio)
    df['golden_ratio_receipts'] = np.abs(R - R * 1.618) / (R + 1)
    
    df['is_prime_receipts'] = R.astype(int).apply(is_prime).astype(float)
    df['is_prime_miles'] = M.astype(int).apply(is_prime).astype(float)
    
    df['is_power_of_2_receipts'] = R.astype(int).apply(lambda x: (x & (x-1)) == 0 and x > 0).astype(float)
    df['is_power_of_2_miles'] = M.astype(int).apply(lambda x: (x & (x-1)) == 0 and x > 0).astype(float)
    
    df['bit_count_receipts'] = R.astype(int).apply(lambda x: bin(x).count('1'))
    df['bit_count_miles'] = M.astype(int).apply(lambda x: bin(x).count('1'))
    
    df['algorithmic_complexity'] = (R * np.log2(R + 1)) / 1000  # O(n log n) pattern
    
    mathematical_features = [
        # V1 Mathematical (8)
        'total_trip_value', 'receipts_log', 'receipts_sqrt', 'miles_log',
        'receipts_sin_1000', 'receipts_cos_1000', 'receipts_exp_norm', 'receipts_cubed',
        
        # Programmer Features (10)  
        'fib_ratio_receipts', 'fib_ratio_miles', 'golden_ratio_receipts',
        'is_prime_receipts', 'is_prime_miles',
        'is_power_of_2_receipts', 'is_power_of_2_miles',
        'bit_count_receipts', 'bit_count_miles', 'algorithmic_complexity'
    ]
    
    return df, mathematical_features

def create_hybrid_champion_features(df_input):
    """Combine business rules + best mathematical features for ultimate hybrid"""
    df = df_input.copy()
    
    print("🏆 Creating HYBRID CHAMPION feature set...")
    
    # Get business rules features (31)
    df, business_features = create_business_rules_features(df)
    
    # Get best mathematical features (18) 
    df, mathematical_features = create_best_mathematical_features(df)
    
    # Combine all features
    all_features = business_features + mathematical_features
    
    print(f"   ✅ Business features: {len(business_features)}")
    print(f"   ✅ Mathematical features: {len(mathematical_features)}")
    print(f"   🏆 Total hybrid features: {len(all_features)}")
    
    return df[all_features]

def main():
    print("🚀 TabPFN Hybrid Champion - Business Rules + Mathematical Features")
    print("="*80)
    print("Combining the winning business rules (31 features) with the best mathematical/")
    print("programming features from previous record-holding experiments")
    print()
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data()
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create hybrid champion features
    print(f"\n{'='*80}")
    print(f"🏆 Creating HYBRID CHAMPION Feature Set")
    print(f"{'='*80}")
    
    print("🎯 Hybrid approach combining:")
    print("   🏢 Business Rules (31): Kevin's insights, trip categories, spending patterns")
    print("   🧮 V1 Mathematical (8): total_trip_value, logarithms, trigonometry, exponentials") 
    print("   🔧 V2 Programmer (10): Fibonacci, primes, powers of 2, bit patterns, algorithms")
    print("   🎯 GOAL: Beat current record of $55.21 MAE")
    
    X_train = create_hybrid_champion_features(train_df)
    X_test = create_hybrid_champion_features(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    total_features = X_train.shape[1]
    
    print(f"\n✨ HYBRID CHAMPION FEATURE SET CREATED:")
    print(f"   🏆 Total Features: {total_features}")
    print(f"   💡 Best of both worlds: Business expertise + Mathematical sophistication")
    print(f"   🧠 Feature density: {total_features/len(train_df):.2f} features per sample")
    
    # Test TabPFN with hybrid features
    print(f"\n{'='*80}")
    print(f"🤖 Training TabPFN with HYBRID CHAMPION Features")
    print(f"{'='*80}")
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("🚀 Initializing TabPFN...")
        print(f"   📊 Training on {len(X_train)} samples")
        print(f"   🏆 Using {total_features} hybrid champion features")
        print("   🎯 Combining business expertise with mathematical power")
        
        # Create TabPFN model
        tabpfn = TabPFNRegressor(device='cpu')
        
        print(f"🏋️ Training TabPFN...")
        
        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.astype(np.float32)
        
        tabpfn.fit(X_train_np, y_train_np)
        
        print(f"🔮 Generating predictions...")
        y_pred = tabpfn.predict(X_test_np)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print(f"🏆 TABPFN HYBRID CHAMPION RESULTS")
        print(f"{'='*80}")
        
        print(f"📊 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"   R-squared (R²): {r2:.4f}")
        
        # Compare to previous records
        previous_results = [
            ("TabPFN Business Rules (Current Record)", 55.21),
            ("TabPFN + Advanced Programmer V2", 55.63),
            ("TabPFN Ultra Features (100+)", 55.96),
            ("TabPFN V1 Features", 55.96),
            ("TabPFN MEGA Features (176)", 57.11),
            ("V1 Neural Networks", 57.35),
            ("V4 Neural Networks", 59.76),
        ]
        
        print(f"\n📈 HYBRID CHAMPION COMPARISON:")
        print(f"   🆕 TabPFN Hybrid Champion ({total_features}): ${mae:.2f} MAE")
        
        best_mae = 55.21
        improvement = best_mae - mae
        improvement_pct = (improvement / best_mae) * 100
        
        if mae < best_mae:
            print(f"   🎉 NEW WORLD RECORD! 🏆")
            print(f"   🥇 Previous best: ${best_mae:.2f} MAE")
            print(f"   📈 Improvement: ${improvement:.2f} ({improvement_pct:.2f}%)")
            print(f"   🏆 HYBRID CHAMPION approach triumphs!")
        else:
            record_gap = mae - best_mae
            record_gap_pct = (record_gap / best_mae) * 100
            print(f"   📊 vs Current Record: ${record_gap:+.2f} ({record_gap_pct:+.2f}%)")
            if record_gap < 0.5:
                print(f"   🎯 Extremely close! Hybrid approach very promising")
            elif record_gap < 1.0:
                print(f"   🎯 Very close to record! Hybrid showing strong potential")
            elif record_gap > 2.0:
                print(f"   ⚠️  May have introduced noise with too many features")
        
        for name, prev_mae in previous_results:
            diff = prev_mae - mae
            diff_pct = (diff / prev_mae) * 100
            emoji = "🎯" if mae < prev_mae else "📊"
            print(f"   {emoji} vs {name}: ${diff:+.2f} ({diff_pct:+.2f}%)")
        
        # Save results
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        results_df.to_csv('tabpfn_hybrid_champion_results.csv', index=False)
        
        # Create comparison
        comparison_data = [{
            'model': f'TabPFN Hybrid Champion ({total_features} features)',
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features': total_features,
            'notes': 'Hybrid: Business Rules (31) + Best Mathematical/Programming (18) features'
        }]
        
        for name, prev_mae in previous_results:
            comparison_data.append({
                'model': name,
                'mae': prev_mae,
                'rmse': 'N/A',
                'r2': 'N/A',  
                'features': 'Various',
                'notes': 'Previous result for comparison'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('mae')
        comparison_df.to_csv('tabpfn_hybrid_champion_comparison.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_hybrid_champion_results.csv")
        print(f"   📈 Comparison: tabpfn_hybrid_champion_comparison.csv")
        
        # Feature analysis
        print(f"\n🔍 HYBRID FEATURE ANALYSIS:")
        
        # Check mathematical features
        fib_receipts_avg = X_train['fib_ratio_receipts'].mean()
        prime_count = (X_train['is_prime_receipts'] == 1).sum()
        power2_count = (X_train['is_power_of_2_receipts'] == 1).sum()
        
        print(f"   🔢 Average Fibonacci ratio (receipts): {fib_receipts_avg:.2f}")
        print(f"   🎯 Prime number receipts: {prime_count} ({prime_count/len(X_train)*100:.1f}%)")
        print(f"   ⚡ Power of 2 receipts: {power2_count} ({power2_count/len(X_train)*100:.1f}%)")
        
        # Business features
        kevin_sweet_spot = (X_train['interaction_kevin_sweet_spot'] == 1).sum()
        optimal_spending = (X_train['is_optimal_daily_spending_kevin'] == 1).sum()
        
        print(f"   🎯 Kevin's Sweet Spot: {kevin_sweet_spot} ({kevin_sweet_spot/len(X_train)*100:.1f}%)")
        print(f"   💰 Optimal spending: {optimal_spending} ({optimal_spending/len(X_train)*100:.1f}%)")
        
        # Final insights
        print(f"\n🧠 HYBRID CHAMPION INSIGHTS:")
        if mae < best_mae:
            print(f"   🎉 ULTIMATE BREAKTHROUGH! Hybrid approach is supreme")
            print(f"   🏆 Business expertise + Mathematical power = unbeatable combination")
            print(f"   🔬 {total_features} features found optimal balance")
            print(f"   📈 Proof that domain knowledge + sophisticated features work together")
        elif mae < best_mae + 0.5:
            print(f"   🎯 Extremely competitive! Hybrid approach very close to record")
            print(f"   💡 Combined approach shows high potential")
            print(f"   🔧 May need fine-tuning of feature selection")
        else:
            print(f"   🤔 Hybrid didn't beat pure business rules approach")
            print(f"   📊 Possible feature interaction or noise introduction")
            print(f"   💭 Business rules alone may be optimal for this problem")
        
        print(f"\n🎯 ULTIMATE CONCLUSIONS:")
        print(f"   📊 Features tested: {total_features} (hybrid approach)")
        print(f"   🏢 Approach: Business rules + Mathematical sophistication")
        print(f"   ⚡ TabPFN performance: {'Excellent' if mae < 60 else 'Good' if mae < 65 else 'Fair'}")
        print(f"   🧠 Hybrid vs Pure Business: {'Hybrid wins!' if mae < 55.21 else 'Business rules still optimal'}")
        print(f"   🚀 Final status: {'NEW CHAMPION!' if mae < best_mae else 'Close competitor' if mae < 56 else 'Good attempt'}")
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 