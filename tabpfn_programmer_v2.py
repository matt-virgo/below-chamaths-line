#!/usr/bin/env python3

"""
TabPFN + Advanced Programmer Features V2
Combining TabPFN with V1 features PLUS 20 sophisticated programmer-specific features
designed specifically for algorithmically generated data by software engineers.
"""

import json
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

def create_v1_ultra_features(df):
    """Create V1's proven comprehensive feature set (58 features)"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # Core derived features
    features_df['miles_per_day'] = M / D
    features_df['receipts_per_day'] = R / D
    
    # Most important validated features
    features_df['total_trip_value'] = D * M * R
    features_df['receipts_log'] = np.log1p(R)
    features_df['receipts_sqrt'] = np.sqrt(R)
    features_df['receipts_squared'] = R ** 2
    features_df['receipts_cubed'] = R ** 3
    
    # Miles transformations
    features_df['miles_log'] = np.log1p(M)
    features_df['miles_sqrt'] = np.sqrt(M)
    features_df['miles_squared'] = M ** 2
    features_df['miles_cubed'] = M ** 3
    
    # Days transformations
    features_df['days_squared'] = D ** 2
    features_df['days_cubed'] = D ** 3
    features_df['days_fourth'] = D ** 4
    
    # Lucky cents feature
    features_df['receipts_cents'] = (R * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(float)
    
    # Comprehensive interactions
    features_df['miles_receipts'] = M * R
    features_df['days_receipts'] = D * R
    features_df['days_miles'] = D * M
    features_df['miles_per_day_squared'] = features_df['miles_per_day'] ** 2
    features_df['receipts_per_day_squared'] = features_df['receipts_per_day'] ** 2
    features_df['miles_receipts_per_day'] = features_df['miles_per_day'] * features_df['receipts_per_day']
    
    # Complex ratio features
    features_df['receipts_to_miles_ratio'] = R / (M + 1)
    features_df['miles_to_days_ratio'] = M / D
    features_df['total_value_per_day'] = features_df['total_trip_value'] / D
    
    # Trigonometric features
    features_df['receipts_sin_1000'] = np.sin(R / 1000)
    features_df['receipts_cos_1000'] = np.cos(R / 1000)
    features_df['receipts_sin_500'] = np.sin(R / 500)
    features_df['receipts_cos_500'] = np.cos(R / 500)
    features_df['miles_sin_500'] = np.sin(M / 500)
    features_df['miles_cos_500'] = np.cos(M / 500)
    features_df['miles_sin_1000'] = np.sin(M / 1000)
    features_df['miles_cos_1000'] = np.cos(M / 1000)
    
    # Exponential features
    features_df['receipts_exp_norm'] = np.exp(R / 2000) - 1
    features_df['miles_exp_norm'] = np.exp(M / 1000) - 1
    features_df['days_exp_norm'] = np.exp(D / 10) - 1
    
    # Polynomial combinations
    features_df['days_miles_receipts'] = D * M * R
    features_df['sqrt_days_miles_receipts'] = np.sqrt(D * M * R)
    features_df['log_days_miles_receipts'] = np.log1p(D * M * R)
    
    # High-order interactions
    features_df['d2_m_r'] = (D ** 2) * M * R
    features_df['d_m2_r'] = D * (M ** 2) * R
    features_df['d_m_r2'] = D * M * (R ** 2)
    
    # Binned features
    features_df['receipts_bin_20'] = pd.cut(R, bins=20, labels=False)
    features_df['miles_bin_20'] = pd.cut(M, bins=20, labels=False)
    features_df['days_bin_10'] = pd.cut(D, bins=10, labels=False)
    
    # Per-day thresholds
    mpd = M / D
    rpd = R / D
    features_df['mpd_low'] = (mpd < 100).astype(float)
    features_df['mpd_med'] = ((mpd >= 100) & (mpd <= 200)).astype(float)
    features_df['mpd_high'] = (mpd > 200).astype(float)
    features_df['rpd_low'] = (rpd < 75).astype(float)
    features_df['rpd_med'] = ((rpd >= 75) & (rpd <= 150)).astype(float)
    features_df['rpd_high'] = (rpd > 150).astype(float)
    
    # Special case indicators
    features_df['is_short_trip'] = (D <= 2).astype(float)
    features_df['is_medium_trip'] = ((D >= 3) & (D <= 7)).astype(float)
    features_df['is_long_trip'] = (D >= 8).astype(float)
    features_df['is_5_day_trip'] = (D == 5).astype(float)
    
    return features_df

def create_advanced_programmer_features(df):
    """Create 20 sophisticated programmer-specific features for software engineer-generated data"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🔧 Creating advanced programmer features...")
    
    # 1. FIBONACCI SOPHISTICATION: Ratios and relationships
    features_df['fib_ratio_receipts'] = R.apply(fibonacci_ratio)
    features_df['fib_ratio_miles'] = M.apply(fibonacci_ratio)
    features_df['golden_ratio_receipts'] = np.abs(R - R * 1.618) / (R + 1)  # Golden ratio deviation
    
    # 2. PRIME NUMBER DENSITY & PATTERNS
    features_df['prime_density'] = R.apply(lambda x: sum(1 for i in range(1, min(int(x), 1000)) if is_prime(i)) / min(x, 1000))
    features_df['is_twin_prime'] = R.apply(lambda x: is_prime(int(x)) and (is_prime(int(x)-2) or is_prime(int(x)+2)))
    
    # 3. BIT PATTERNS & BINARY REPRESENTATION
    features_df['bit_count_receipts'] = R.astype(int).apply(lambda x: bin(x).count('1'))
    features_df['bit_count_miles'] = M.astype(int).apply(lambda x: bin(x).count('1'))
    features_df['hamming_weight'] = (features_df['bit_count_receipts'] + features_df['bit_count_miles']) / 2
    
    # 4. HASH-LIKE BEHAVIORS (simple hash patterns)
    features_df['pseudo_hash'] = ((R * 31 + M * 17 + D * 13) % 997) / 997  # Simple polynomial hash
    features_df['checksum_like'] = ((R.astype(int) + M.astype(int) + D) % 256) / 256
    
    # 5. ALGORITHM COMPLEXITY PATTERNS
    features_df['log_complexity'] = np.log2(R * M + 1)  # O(log n) patterns
    features_df['nlogn_complexity'] = (R * np.log2(R + 1)) / 10000  # O(n log n) patterns
    features_df['quadratic_complexity'] = (R * R) / (M * D + 1)  # O(n²) patterns
    
    # 6. SOFTWARE CONSTANTS (pi, e, sqrt(2) relationships)
    features_df['pi_relationship'] = np.abs(R - R * np.pi) / (R * np.pi + 1)
    features_df['e_relationship'] = np.abs(M - M * np.e) / (M * np.e + 1)
    features_df['sqrt2_relationship'] = np.abs(D - D * np.sqrt(2)) / (D * np.sqrt(2) + 1)
    
    # 7. MODULAR ARITHMETIC PATTERNS
    features_df['mod_pattern_7'] = ((R * M * D) % 7) / 7
    features_df['mod_pattern_11'] = ((R + M + D) % 11) / 11
    features_df['mod_pattern_13'] = ((R * M + D) % 13) / 13
    
    # 8. BOUNDARY CONDITION PATTERNS (common in testing)
    features_df['is_boundary_receipts'] = ((R % 100 == 0) | (R % 100 == 1) | (R % 100 == 99)).astype(float)
    features_df['is_power_boundary'] = ((M.astype(int) & (M.astype(int) - 1)) == 0).astype(float)  # Power of 2 check
    
    print(f"   ✨ Created {len([col for col in features_df.columns if col not in df.columns])} advanced programmer features")
    
    return features_df

def create_combined_features(df):
    """Combine V1 features with advanced programmer features"""
    # Start with V1 features
    combined_df = create_v1_ultra_features(df)
    
    # Add programmer features
    programmer_df = create_advanced_programmer_features(df)
    
    # Merge the new programmer features
    for col in programmer_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = programmer_df[col]
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in combined_df.columns if col != 'reimbursement']
    
    return combined_df[feature_cols]

def main():
    print("🚀 TabPFN + Advanced Programmer Features V2")
    print("="*70)
    print("Testing TabPFN with V1 features + 20 sophisticated programmer-specific features")
    print("Designed for software engineer-generated algorithmic data")
    print()
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Dataset overview:")
    print(f"   📊 Training samples: {len(train_df)}")
    print(f"   📊 Test samples: {len(test_df)}")
    
    # Create combined feature set
    print(f"\n{'='*70}")
    print(f"🔧 Creating Combined Feature Set")
    print(f"{'='*70}")
    
    print("Creating V1 comprehensive features (58 features)...")
    X_train = create_combined_features(train_df)
    X_test = create_combined_features(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    total_features = X_train.shape[1]
    v1_features = 58
    programmer_features = total_features - v1_features
    
    print(f"✨ Feature summary:")
    print(f"   📊 V1 Features: {v1_features}")
    print(f"   🔧 Advanced Programmer Features: {programmer_features}")
    print(f"   📈 Total Features: {total_features}")
    
    # Test TabPFN
    print(f"\n{'='*70}")
    print(f"🤖 Training TabPFN with Combined Features")
    print(f"{'='*70}")
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("🚀 Initializing TabPFN...")
        print("   📱 Using CPU mode for compatibility")
        
        # Create TabPFN model
        tabpfn = TabPFNRegressor(device='cpu')
        
        print(f"🏋️ Training on {len(X_train)} samples with {total_features} features...")
        tabpfn.fit(X_train.values, y_train)
        
        print(f"🔮 Making predictions on {len(X_test)} test samples...")
        y_pred = tabpfn.predict(X_test.values)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print(f"🏆 TabPFN + Advanced Programmer Features RESULTS")
        print(f"{'='*80}")
        
        print(f"📊 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"   R-squared (R²): {r2:.4f}")
        
        # Compare to previous results
        previous_results = [
            ("TabPFN V1 Features (Current Champion)", 55.96),
            ("V1 Neural Networks", 57.35),
            ("TabPFN + V2 Programmer Features", 63.72),  # Previous programmer attempt
            ("V4 Neural Networks", 59.76),
            ("XGBoost V1 Features", 63.50),
        ]
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   🆕 TabPFN + Advanced Programmer: ${mae:.2f} MAE")
        
        best_mae = 55.96
        improvement = best_mae - mae
        improvement_pct = (improvement / best_mae) * 100
        
        if mae < best_mae:
            print(f"   🎉 NEW WORLD RECORD!")
            print(f"   🏆 Previous best: ${best_mae:.2f} MAE")
            print(f"   📈 Improvement: ${improvement:.2f} ({improvement_pct:.2f}%)")
        else:
            record_gap = mae - best_mae
            print(f"   📊 vs Current Record: ${record_gap:+.2f}")
        
        for name, prev_mae in previous_results:
            diff = prev_mae - mae
            diff_pct = (diff / prev_mae) * 100
            emoji = "🎯" if mae < prev_mae else "📊"
            print(f"   {emoji} vs {name}: ${diff:+.2f} ({diff_pct:+.2f}%)")
        
        # Analyze feature importance through prediction variance
        print(f"\n🔬 ADVANCED PROGRAMMER FEATURES ANALYSIS:")
        
        # Test prediction stability by feature group
        print("Testing feature group contributions...")
        
        # V1 only prediction
        X_test_v1_only = create_v1_ultra_features(test_df)
        v1_feature_cols = [col for col in X_test_v1_only.columns if col != 'reimbursement']
        X_test_v1_only = X_test_v1_only[v1_feature_cols]
        
        tabpfn_v1 = TabPFNRegressor(device='cpu')
        X_train_v1_only = create_v1_ultra_features(train_df)
        X_train_v1_only = X_train_v1_only[v1_feature_cols]
        
        tabpfn_v1.fit(X_train_v1_only.values, y_train)
        y_pred_v1_only = tabpfn_v1.predict(X_test_v1_only.values)
        mae_v1_only = mean_absolute_error(y_test, y_pred_v1_only)
        
        programmer_contribution = mae_v1_only - mae
        print(f"   📊 V1 Features Only: ${mae_v1_only:.2f} MAE")
        print(f"   🔧 Programmer Features Contribution: ${programmer_contribution:+.2f} MAE")
        
        if programmer_contribution > 0:
            print(f"   ✅ Advanced programmer features improved performance!")
            print(f"   🎯 TabPFN successfully leveraged algorithmic patterns")
        else:
            print(f"   🤔 Advanced programmer features didn't help")
            print(f"   💭 TabPFN may already capture these patterns implicitly")
        
        # Save results
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        results_df.to_csv('tabpfn_programmer_v2_results.csv', index=False)
        
        # Create detailed comparison
        comparison_data = []
        comparison_data.append({
            'model': 'TabPFN + Advanced Programmer V2',
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features': total_features,
            'notes': f'{v1_features} V1 + {programmer_features} advanced programmer features'
        })
        
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
        comparison_df.to_csv('tabpfn_programmer_v2_comparison.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_programmer_v2_results.csv")
        print(f"   📈 Comparison: tabpfn_programmer_v2_comparison.csv")
        
        # Final insights
        print(f"\n🧠 KEY INSIGHTS:")
        if mae < best_mae:
            print(f"   🎉 BREAKTHROUGH! Advanced programmer features + TabPFN = new record")
            print(f"   🔧 Software engineering patterns add value to foundation models")
            print(f"   🚀 Algorithmic data generation creates exploitable patterns")
        elif programmer_contribution > 0:
            print(f"   ✅ Advanced programmer features provided modest improvement")
            print(f"   🎯 TabPFN can leverage algorithmic patterns better than neural networks")
            print(f"   🔬 Foundation models + domain-specific features = powerful combination")
        else:
            print(f"   🤔 Advanced features didn't improve performance")
            print(f"   💭 TabPFN may already capture programmer patterns implicitly")
            print(f"   📊 V1 comprehensive approach remains best for this dataset")
        
        print(f"   🔍 TabPFN consistently outperforms traditional ML on this task")
        print(f"   ⚡ Zero hyperparameter tuning required")
        print(f"   🏗️ Feature engineering still valuable for foundation models")
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 