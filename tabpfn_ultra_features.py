#!/usr/bin/env python3

"""
TabPFN Ultra Features - 100+ Feature Experiment
Pushing the boundaries with comprehensive feature engineering for TabPFN
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
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
    """Create 21 sophisticated programmer-specific features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # 1. FIBONACCI SOPHISTICATION: Ratios and relationships
    features_df['fib_ratio_receipts'] = R.apply(fibonacci_ratio)
    features_df['fib_ratio_miles'] = M.apply(fibonacci_ratio)
    features_df['golden_ratio_receipts'] = np.abs(R - R * 1.618) / (R + 1)
    
    # 2. PRIME NUMBER DENSITY & PATTERNS
    features_df['prime_density'] = R.apply(lambda x: sum(1 for i in range(1, min(int(x), 1000)) if is_prime(i)) / min(x, 1000))
    features_df['is_twin_prime'] = R.apply(lambda x: is_prime(int(x)) and (is_prime(int(x)-2) or is_prime(int(x)+2)))
    
    # 3. BIT PATTERNS & BINARY REPRESENTATION
    features_df['bit_count_receipts'] = R.astype(int).apply(lambda x: bin(x).count('1'))
    features_df['bit_count_miles'] = M.astype(int).apply(lambda x: bin(x).count('1'))
    features_df['hamming_weight'] = (features_df['bit_count_receipts'] + features_df['bit_count_miles']) / 2
    
    # 4. HASH-LIKE BEHAVIORS
    features_df['pseudo_hash'] = ((R * 31 + M * 17 + D * 13) % 997) / 997
    features_df['checksum_like'] = ((R.astype(int) + M.astype(int) + D) % 256) / 256
    
    # 5. ALGORITHM COMPLEXITY PATTERNS
    features_df['log_complexity'] = np.log2(R * M + 1)
    features_df['nlogn_complexity'] = (R * np.log2(R + 1)) / 10000
    features_df['quadratic_complexity'] = (R * R) / (M * D + 1)
    
    # 6. SOFTWARE CONSTANTS
    features_df['pi_relationship'] = np.abs(R - R * np.pi) / (R * np.pi + 1)
    features_df['e_relationship'] = np.abs(M - M * np.e) / (M * np.e + 1)
    features_df['sqrt2_relationship'] = np.abs(D - D * np.sqrt(2)) / (D * np.sqrt(2) + 1)
    
    # 7. MODULAR ARITHMETIC PATTERNS
    features_df['mod_pattern_7'] = ((R * M * D) % 7) / 7
    features_df['mod_pattern_11'] = ((R + M + D) % 11) / 11
    features_df['mod_pattern_13'] = ((R * M + D) % 13) / 13
    
    # 8. BOUNDARY CONDITION PATTERNS
    features_df['is_boundary_receipts'] = ((R % 100 == 0) | (R % 100 == 1) | (R % 100 == 99)).astype(float)
    features_df['is_power_boundary'] = ((M.astype(int) & (M.astype(int) - 1)) == 0).astype(float)
    
    return features_df

def create_ultra_mathematical_features(df):
    """Create 25+ ultra-sophisticated mathematical features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🧮 Creating ultra-mathematical features...")
    
    # 1. HIGHER ORDER POLYNOMIALS
    features_df['receipts_fifth'] = R ** 5
    features_df['miles_fifth'] = M ** 5
    features_df['days_fifth'] = D ** 5
    features_df['receipts_sixth'] = R ** 6
    
    # 2. RECIPROCAL AND INVERSE FEATURES
    features_df['receipts_reciprocal'] = 1 / (R + 1)
    features_df['miles_reciprocal'] = 1 / (M + 1)
    features_df['days_reciprocal'] = 1 / (D + 1)
    
    # 3. ADVANCED TRIGONOMETRIC PATTERNS
    features_df['receipts_tan_1000'] = np.tan(R / 1000)
    features_df['miles_tan_500'] = np.tan(M / 500)
    features_df['receipts_sin_250'] = np.sin(R / 250)
    features_df['receipts_cos_250'] = np.cos(R / 250)
    features_df['miles_sin_250'] = np.sin(M / 250)
    features_df['days_sin_10'] = np.sin(D / 10)
    features_df['days_cos_10'] = np.cos(D / 10)
    
    # 4. STATISTICAL FEATURES
    all_receipts = np.concatenate([R.values] * 3)  # Replicate for percentile calc
    features_df['receipts_percentile'] = R.apply(lambda x: stats.percentileofscore(all_receipts, x) / 100)
    features_df['receipts_zscore'] = (R - R.mean()) / (R.std() + 1e-8)
    features_df['miles_zscore'] = (M - M.mean()) / (M.std() + 1e-8)
    
    # 5. LOGARITHMIC VARIANTS
    features_df['receipts_log10'] = np.log10(R + 1)
    features_df['miles_log10'] = np.log10(M + 1)
    features_df['receipts_log2'] = np.log2(R + 1)
    features_df['miles_log2'] = np.log2(M + 1)
    features_df['total_log_product'] = np.log1p(R) * np.log1p(M)  # Use direct calculation
    
    # 6. HYPERBOLIC FUNCTIONS
    features_df['receipts_sinh'] = np.sinh(R / 5000)
    features_df['receipts_cosh'] = np.cosh(R / 5000)
    features_df['miles_tanh'] = np.tanh(M / 1000)
    
    # 7. POLYNOMIAL INTERACTIONS (3-way, 4-way)
    features_df['d3_m_r'] = (D ** 3) * M * R
    features_df['d_m3_r'] = D * (M ** 3) * R
    features_df['d_m_r3'] = D * M * (R ** 3)
    features_df['d2_m2_r'] = (D ** 2) * (M ** 2) * R
    features_df['d2_m_r2'] = (D ** 2) * M * (R ** 2)
    features_df['d_m2_r2'] = D * (M ** 2) * (R ** 2)
    features_df['d2_m2_r2'] = (D ** 2) * (M ** 2) * (R ** 2)
    
    return features_df

def create_business_intelligence_features(df):
    """Create 15+ business and economic intelligence features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   💼 Creating business intelligence features...")
    
    # 1. EFFICIENCY METRICS
    features_df['cost_efficiency'] = R / (M * D + 1)  # Cost per mile-day
    features_df['travel_intensity'] = M / (D ** 2 + 1)  # Miles per day squared
    features_df['spending_velocity'] = R / (D ** 0.5 + 1)  # Spending rate
    
    # 2. BUSINESS CATEGORY FEATURES
    mpd = M / D
    rpd = R / D
    features_df['is_local_business'] = ((mpd < 50) & (rpd < 100)).astype(float)
    features_df['is_road_warrior'] = ((mpd > 300) & (D > 3)).astype(float)
    features_df['is_conference_trip'] = ((D >= 3) & (D <= 5) & (rpd > 150)).astype(float)
    features_df['is_sales_trip'] = ((mpd > 100) & (mpd < 300) & (rpd < 120)).astype(float)
    
    # 3. ECONOMIC EFFICIENCY INDICATORS
    features_df['miles_per_dollar'] = M / (R + 1)
    features_df['dollar_per_mile'] = R / (M + 1)
    features_df['time_value_ratio'] = (M * R) / (D ** 2 + 1)
    
    # 4. EXPENSE PATTERN RECOGNITION
    features_df['low_cost_long_trip'] = ((D > 7) & (rpd < 75)).astype(float)
    features_df['high_cost_short_trip'] = ((D <= 3) & (rpd > 150)).astype(float)
    features_df['balanced_trip'] = ((rpd >= 75) & (rpd <= 150) & (mpd >= 100) & (mpd <= 200)).astype(float)
    
    # 5. ADVANCED RATIOS
    features_df['geometric_mean_activity'] = np.sqrt(M * R)
    features_df['harmonic_mean_per_day'] = 2 / ((1/(M/D + 1)) + (1/(R/D + 1)))
    features_df['activity_asymmetry'] = np.abs(np.log((M/D + 1) / (R/D + 1)))
    
    return features_df

def create_combined_ultra_features(df):
    """Combine all feature sets for 100+ total features"""
    # Start with V1 features (58)
    combined_df = create_v1_ultra_features(df)
    
    # Add advanced programmer features (21)
    programmer_df = create_advanced_programmer_features(df)
    for col in programmer_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = programmer_df[col]
    
    # Add ultra-mathematical features (25+)
    math_df = create_ultra_mathematical_features(df)
    for col in math_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = math_df[col]
    
    # Add business intelligence features (15+)
    business_df = create_business_intelligence_features(df)
    for col in business_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = business_df[col]
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in combined_df.columns if col != 'reimbursement']
    
    return combined_df[feature_cols]

def main():
    print("🚀 TabPFN Ultra Features - 100+ Feature Experiment")
    print("="*70)
    print("Pushing the boundaries of feature engineering with TabPFN")
    print()
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data()
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create ultra feature set
    print(f"\n{'='*70}")
    print(f"🔧 Creating Ultra-Comprehensive Feature Set")
    print(f"{'='*70}")
    
    print("🎯 Feature categories:")
    print("   📊 V1 Core Features: 58")
    print("   🔧 Advanced Programmer Features: 21") 
    print("   🧮 Ultra-Mathematical Features: 25+")
    print("   💼 Business Intelligence Features: 15+")
    print("   🎯 Target: 100+ total features")
    
    X_train = create_combined_ultra_features(train_df)
    X_test = create_combined_ultra_features(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    total_features = X_train.shape[1]
    
    print(f"\n✨ ULTRA FEATURE SET CREATED:")
    print(f"   📈 Total Features: {total_features}")
    print(f"   🎯 Target Achieved: {'YES! 🎉' if total_features >= 100 else 'NO, need more'}")
    
    # Test TabPFN with ultra features
    print(f"\n{'='*70}")
    print(f"🤖 Training TabPFN with Ultra Features")
    print(f"{'='*70}")
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("🚀 Initializing TabPFN...")
        print(f"   📊 Training on {len(X_train)} samples")
        print(f"   🔧 Using {total_features} ultra-engineered features")
        print("   ⚡ This may take a while with 100+ features...")
        
        # Create TabPFN model
        tabpfn = TabPFNRegressor(device='cpu')
        
        print(f"🏋️ Training TabPFN...")
        tabpfn.fit(X_train.values, y_train)
        
        print(f"🔮 Generating predictions...")
        y_pred = tabpfn.predict(X_test.values)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print(f"🏆 TABPFN ULTRA FEATURES RESULTS")
        print(f"{'='*80}")
        
        print(f"📊 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"   R-squared (R²): {r2:.4f}")
        
        # Compare to previous records
        previous_results = [
            ("TabPFN + Advanced Programmer V2 (Previous Record)", 55.63),
            ("TabPFN V1 Features", 55.96),
            ("V1 Neural Networks", 57.35),
            ("V4 Neural Networks", 59.76),
        ]
        
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"   🆕 TabPFN Ultra Features ({total_features}): ${mae:.2f} MAE")
        
        best_mae = 55.63
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
        
        # Save results
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        results_df.to_csv('tabpfn_ultra_features_results.csv', index=False)
        
        # Create comparison
        comparison_data = [{
            'model': f'TabPFN Ultra Features ({total_features} features)',
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features': total_features,
            'notes': 'V1 + Advanced Programmer + Ultra Math + Business Intelligence'
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
        comparison_df.to_csv('tabpfn_ultra_features_comparison.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_ultra_features_results.csv")
        print(f"   📈 Comparison: tabpfn_ultra_features_comparison.csv")
        
        # Final insights
        print(f"\n🧠 ULTRA FEATURES INSIGHTS:")
        if mae < best_mae:
            print(f"   🎉 BREAKTHROUGH! {total_features} features achieved new record")
            print(f"   🔬 Ultra feature engineering pays off with TabPFN")
            print(f"   📈 More sophisticated patterns = better predictions")
        else:
            print(f"   🤔 {total_features} features didn't improve performance")
            print(f"   💭 Possible feature saturation or noise introduction")
            print(f"   📊 Previous {len(previous_results[0][0].split())} feature approach may be optimal")
        
        print(f"   🎯 Feature Engineering Conclusion:")
        print(f"   📊 Total features tested: {total_features}")
        print(f"   🔧 Mathematical features: Advanced polynomials, trig, hyperbolic")
        print(f"   💼 Business features: Efficiency metrics, category detection")
        print(f"   ⚡ TabPFN handles high-dimensional feature spaces well")
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 