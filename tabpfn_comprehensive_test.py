#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
    
    # Lucky cents feature (validated as important)
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
    
    # Trigonometric features (for cyclical patterns)
    features_df['receipts_sin_1000'] = np.sin(R / 1000)
    features_df['receipts_cos_1000'] = np.cos(R / 1000)
    features_df['receipts_sin_500'] = np.sin(R / 500)
    features_df['receipts_cos_500'] = np.cos(R / 500)
    features_df['miles_sin_500'] = np.sin(M / 500)
    features_df['miles_cos_500'] = np.cos(M / 500)
    features_df['miles_sin_1000'] = np.sin(M / 1000)
    features_df['miles_cos_1000'] = np.cos(M / 1000)
    
    # Exponential features (normalized)
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
    
    # Binned features (for threshold detection)
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
    
    # Remove target if present
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    
    print(f"V1 ultra feature set created: {len(feature_cols)} comprehensive features")
    return features_df[feature_cols]

def main():
    print("⚡ TabPFN Foundation Model - Comprehensive Test")
    print("="*70)
    print("Testing state-of-the-art foundation model for tabular data")
    print()
    
    # First, try to import TabPFN
    try:
        from tabpfn import TabPFNRegressor
        print("✅ TabPFN successfully imported!")
    except ImportError:
        print("❌ TabPFN not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tabpfn"])
        try:
            from tabpfn import TabPFNRegressor
            print("✅ TabPFN installed and imported successfully!")
        except ImportError:
            print("❌ Failed to install TabPFN. Please install manually: pip install tabpfn")
            return
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Dataset size: {len(train_df)} train + {len(test_df)} test = {len(train_df) + len(test_df)} total")
    print("✅ Perfect size for TabPFN (optimized for datasets up to 10,000 rows)")
    
    # Test configurations
    test_configs = [
        {
            'name': 'TabPFN_Raw_Features',
            'description': 'TabPFN with only 3 raw input features',
            'features': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount'],
            'use_v1_features': False
        },
        {
            'name': 'TabPFN_V1_Engineered',
            'description': 'TabPFN with V1\'s 58 engineered features',
            'features': None,  # Will be created
            'use_v1_features': True
        }
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"⚡ Testing: {config['name']}")
        print(f"📝 {config['description']}")
        print(f"{'='*70}")
        
        # Prepare features
        if config['use_v1_features']:
            print("Creating V1's comprehensive engineered features...")
            X_train = create_v1_ultra_features(train_df)
            X_test = create_v1_ultra_features(test_df)
            print(f"✨ Using {X_train.shape[1]} V1 engineered features")
        else:
            X_train = train_df[config['features']]
            X_test = test_df[config['features']]
            print(f"✨ Using {len(config['features'])} raw features:")
            for i, feature in enumerate(config['features'], 1):
                print(f"   {i}. {feature}")
        
        y_train = train_df['reimbursement'].values
        y_test = test_df['reimbursement'].values
        
        print(f"\nTraining TabPFN model...")
        print("🚀 TabPFN advantages:")
        print("   • Foundation model pre-trained on diverse tabular datasets")
        print("   • No hyperparameter tuning required")
        print("   • Excellent performance on small datasets")
        print("   • Handles complex patterns automatically")
        
        # Create and train TabPFN regressor
        try:
            regressor = TabPFNRegressor(device='cpu')  # Use CPU for compatibility
            print("   📱 Using CPU mode for broad compatibility")
            
            # Fit the model
            print("   🎯 Training TabPFN...")
            regressor.fit(X_train, y_train)
            print("   ✅ Training completed!")
            
            # Make predictions
            print("   🔮 Making predictions...")
            train_pred = regressor.predict(X_train)
            test_pred = regressor.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Precision metrics
            exact_matches = np.sum(np.abs(y_test - test_pred) < 0.01)
            close_matches_1 = np.sum(np.abs(y_test - test_pred) < 1.0)
            close_matches_5 = np.sum(np.abs(y_test - test_pred) < 5.0)
            close_matches_10 = np.sum(np.abs(y_test - test_pred) < 10.0)
            
            print(f"\n🎯 {config['name']} RESULTS:")
            print(f"   Train MAE: ${train_mae:.2f}")
            print(f"   Test MAE:  ${test_mae:.2f}")
            print(f"   Train R²:  {train_r2:.6f}")
            print(f"   Test R²:   {test_r2:.6f}")
            print(f"   Exact matches (±$0.01): {exact_matches}")
            print(f"   Close matches (±$1.00): {close_matches_1}")
            print(f"   Close matches (±$5.00): {close_matches_5}")
            print(f"   Close matches (±$10.00): {close_matches_10}")
            
            results = {
                'name': config['name'],
                'description': config['description'],
                'features_count': X_train.shape[1],
                'train_mae': train_mae,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'exact_matches': exact_matches,
                'close_matches_1': close_matches_1,
                'close_matches_5': close_matches_5,
                'close_matches_10': close_matches_10,
                'predictions': test_pred,
                'success': True
            }
            
            all_results.append(results)
            
            # Save results
            tabpfn_results = pd.DataFrame({
                'trip_duration_days': test_df['trip_duration_days'],
                'miles_traveled': test_df['miles_traveled'],
                'total_receipts_amount': test_df['total_receipts_amount'],
                'actual_reimbursement': test_df['reimbursement'],
                'tabpfn_prediction': test_pred,
                'error': test_df['reimbursement'] - test_pred,
                'abs_error': np.abs(test_df['reimbursement'] - test_pred)
            })
            
            results_file = f"{config['name'].lower()}_results.csv"
            tabpfn_results.to_csv(results_file, index=False)
            print(f"   💾 Results saved to: {results_file}")
            
        except Exception as e:
            print(f"   ❌ Error with {config['name']}: {str(e)}")
            results = {
                'name': config['name'],
                'description': config['description'],
                'features_count': X_train.shape[1],
                'test_mae': float('inf'),
                'success': False,
                'error': str(e)
            }
            all_results.append(results)
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"⚡ TabPFN FOUNDATION MODEL FINAL RESULTS:")
    print(f"{'='*80}")
    
    # Filter successful results and sort by test MAE
    successful_results = [r for r in all_results if r.get('success', False)]
    if successful_results:
        sorted_results = sorted(successful_results, key=lambda x: x['test_mae'])
        
        for i, result in enumerate(sorted_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            print(f"{rank_emoji} {result['name']:<25} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f} | Features: {result['features_count']:2d}")
        
        best_tabpfn = sorted_results[0]
        
        print(f"\n🎉 BEST TabPFN MODEL: {best_tabpfn['name']}")
        print(f"   🎯 Test MAE: ${best_tabpfn['test_mae']:.2f}")
        print(f"   📊 Test R²: {best_tabpfn['test_r2']:.6f}")
        print(f"   🔧 Features: {best_tabpfn['features_count']}")
        print(f"   📝 {best_tabpfn['description']}")
        
        print(f"\n📈 COMPREHENSIVE COMPARISON:")
        print(f"   🏆 V1 Neural (58 features):      $57.35 MAE")
        print(f"   🥈 V4 Neural (58 features):      $59.76 MAE")
        print(f"   🥉 Ensemble V1:                  $62.82 MAE")
        print(f"   4️⃣  XGBoost V1 (58 features):    $63.50 MAE")
        print(f"   5️⃣  Raw XGBoost (3 features):    $79.25 MAE")
        print(f"   ⚡ TabPFN Best:                   ${best_tabpfn['test_mae']:.2f} MAE")
        
        if best_tabpfn['test_mae'] < 57.35:
            improvement = 57.35 - best_tabpfn['test_mae']
            print(f"   🎉 NEW WORLD RECORD! TabPFN beats V1 by ${improvement:.2f} ({improvement/57.35*100:.1f}% improvement)")
        elif best_tabpfn['test_mae'] < 59.76:
            improvement = 59.76 - best_tabpfn['test_mae']
            print(f"   🚀 TabPFN beats V4! ${improvement:.2f} improvement")
        elif best_tabpfn['test_mae'] < 63.50:
            improvement = 63.50 - best_tabpfn['test_mae']
            print(f"   ✅ TabPFN beats XGBoost! ${improvement:.2f} improvement")
        else:
            diff = best_tabpfn['test_mae'] - 57.35
            print(f"   📊 TabPFN is ${diff:.2f} behind V1 baseline")
        
        print(f"\n🧠 TabPFN INSIGHTS:")
        print(f"   • Foundation model trained on diverse tabular datasets")
        print(f"   • Zero hyperparameter tuning required")
        print(f"   • Excellent performance on small datasets like ours")
        print(f"   • Transformer architecture designed for tabular data")
        print(f"   • Can handle both raw and engineered features effectively")
        
        if len(successful_results) > 1:
            raw_result = next((r for r in successful_results if 'Raw' in r['name']), None)
            eng_result = next((r for r in successful_results if 'Engineered' in r['name']), None)
            
            if raw_result and eng_result:
                print(f"\n🔬 RAW vs ENGINEERED FEATURES ANALYSIS:")
                if raw_result['test_mae'] < eng_result['test_mae']:
                    improvement = eng_result['test_mae'] - raw_result['test_mae']
                    print(f"   ✨ Raw features WIN! ${improvement:.2f} better than engineered")
                    print(f"   💡 TabPFN's foundation model handles raw features excellently")
                else:
                    improvement = raw_result['test_mae'] - eng_result['test_mae']
                    print(f"   🏗️  Engineered features WIN! ${improvement:.2f} better than raw")
                    print(f"   💡 Feature engineering still valuable for TabPFN")
    
    else:
        print("❌ No successful TabPFN results. Please check installation and requirements.")
        print("🔧 Try: pip install tabpfn")
        print("🔧 Requires Python 3.9+")

if __name__ == "__main__":
    main() 