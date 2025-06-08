#!/usr/bin/env python3

"""
TabPFN Post-Hoc Ensemble Analysis
Using AutoTabPFNRegressor to automatically create and ensemble multiple TabPFN models
WARNING: This may run slowly on CPU-only systems and is computationally intensive.
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
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    
    return features_df[feature_cols]

def main():
    print("🚀 TabPFN Post-Hoc Ensemble Analysis")
    print("="*70)
    print("Using AutoTabPFNRegressor to automatically ensemble multiple TabPFN models")
    print("WARNING: This is computationally intensive and may take several minutes")
    print()
    
    # First, try to import TabPFN Extensions
    try:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
        print("✅ TabPFN Extensions post-hoc ensembles successfully imported!")
    except ImportError:
        print("❌ TabPFN Extensions not found. Installing...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"])
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
            print("✅ TabPFN Extensions installed and imported successfully!")
        except Exception as e:
            print(f"❌ Failed to install TabPFN Extensions: {str(e)}")
            print("🔧 Manual installation: pip install 'tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git'")
            return
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Dataset overview:")
    print(f"   📊 Training samples: {len(train_df)}")
    print(f"   📊 Test samples: {len(test_df)}")
    
    # Create V1's proven feature set
    print("\nCreating V1's comprehensive engineered features...")
    X_train_engineered = create_v1_ultra_features(train_df)
    X_test_engineered = create_v1_ultra_features(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"✨ Using {X_train_engineered.shape[1]} V1 engineered features")
    
    # Test different time budgets for ensemble training
    time_budgets = [
        {"name": "Quick Ensemble", "max_time": 60 * 3},      # 3 minutes
        {"name": "Standard Ensemble", "max_time": 60 * 5},   # 5 minutes  
        {"name": "Extended Ensemble", "max_time": 60 * 8},   # 8 minutes
    ]
    
    all_results = []
    
    for config in time_budgets:
        print(f"\n{'='*70}")
        print(f"🎯 Training: {config['name']}")
        print(f"📝 Time budget: {config['max_time'] // 60} minutes")
        print(f"{'='*70}")
        
        try:
            print(f"🚀 Initializing AutoTabPFNRegressor...")
            print(f"   ⏱️  Max training time: {config['max_time'] // 60} minutes")
            print(f"   🖥️  Using CPU mode for compatibility")
            
            # Create AutoTabPFNRegressor
            regressor = AutoTabPFNRegressor(
                max_time=config['max_time'],
                device='cpu'  # Use CPU for compatibility
            )
            
            print(f"🏋️ Training ensemble on {len(X_train_engineered)} samples...")
            print("   This will train multiple TabPFN models and automatically ensemble them")
            print("   ⏳ Please be patient, this may take several minutes...")
            
            # Fit the ensemble
            regressor.fit(X_train_engineered.values, y_train)
            
            print(f"✅ Training completed!")
            
            # Make predictions
            print(f"🔮 Making predictions on {len(X_test_engineered)} test samples...")
            y_pred = regressor.predict(X_test_engineered.values)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\n🎯 {config['name']} RESULTS:")
            print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
            print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
            print(f"   R-squared (R²): {r2:.4f}")
            
            # Compare to previous best results
            previous_best_mae = 55.96
            improvement = previous_best_mae - mae
            improvement_pct = (improvement / previous_best_mae) * 100
            
            if mae < previous_best_mae:
                print(f"   🏆 NEW RECORD! Improved by ${improvement:.2f} ({improvement_pct:.2f}%)")
            else:
                print(f"   📊 vs Previous Best ($55.96): ${mae - previous_best_mae:+.2f} ({(mae/previous_best_mae - 1)*100:+.2f}%)")
            
            # Save individual results
            results_df = pd.DataFrame({
                'actual': y_test,
                'predicted': y_pred,
                'absolute_error': np.abs(y_test - y_pred)
            })
            
            filename = f"tabpfn_ensemble_{config['name'].lower().replace(' ', '_')}_results.csv"
            results_df.to_csv(filename, index=False)
            print(f"   💾 Results saved to: {filename}")
            
            # Store results
            result = {
                'name': config['name'],
                'time_budget_minutes': config['max_time'] // 60,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'improvement_vs_best': improvement,
                'improvement_pct': improvement_pct,
                'predictions': y_pred,
                'success': True
            }
            
            all_results.append(result)
            
        except Exception as e:
            print(f"   ❌ Error with {config['name']}: {str(e)}")
            print(f"   This might be due to memory constraints or compatibility issues")
            
            result = {
                'name': config['name'],
                'time_budget_minutes': config['max_time'] // 60,
                'success': False,
                'error': str(e)
            }
            all_results.append(result)
    
    # Overall analysis
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) > 0:
        print(f"\n{'='*80}")
        print(f"🏆 POST-HOC ENSEMBLE PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        print(f"Performance comparison:")
        print(f"   🥇 Previous Best (TabPFN V1): $55.96 MAE")
        
        # Sort by performance
        successful_results.sort(key=lambda x: x['mae'])
        
        for i, result in enumerate(successful_results):
            rank_emoji = ["🥇", "🥈", "🥉"][min(i, 2)]
            print(f"   {rank_emoji} {result['name']} ({result['time_budget_minutes']}min): "
                  f"${result['mae']:.2f} MAE ({result['improvement_pct']:+.2f}%)")
        
        # Best performing ensemble
        best_ensemble = successful_results[0]
        
        if best_ensemble['mae'] < 55.96:
            print(f"\n🎉 BREAKTHROUGH ACHIEVED!")
            print(f"   Best ensemble: {best_ensemble['name']}")
            print(f"   New record: ${best_ensemble['mae']:.2f} MAE")
            print(f"   Improvement: ${best_ensemble['improvement']:.2f} ({best_ensemble['improvement_pct']:.2f}%)")
        else:
            print(f"\n📊 Ensemble Performance Analysis:")
            print(f"   • Best ensemble MAE: ${best_ensemble['mae']:.2f}")
            print(f"   • Previous best was still superior by ${best_ensemble['mae'] - 55.96:.2f}")
            print(f"   • Post-hoc ensembles may need more time or different configuration")
        
        # Analysis insights
        print(f"\n🧠 ENSEMBLE INSIGHTS:")
        print(f"   • AutoTabPFNRegressor trains multiple TabPFN models automatically")
        print(f"   • Longer training times may lead to better ensemble diversity")
        print(f"   • Post-hoc ensembles can be more robust than single models")
        print(f"   • Performance depends on ensemble composition and weighting")
        
        # Save comprehensive results
        comparison_df = pd.DataFrame([
            {
                'model': 'Previous Best (TabPFN V1)',
                'mae': 55.96,
                'time_budget_minutes': 'N/A',
                'notes': 'Single TabPFN model with V1 features'
            }
        ])
        
        for result in successful_results:
            comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                'model': f"{result['name']} Ensemble",
                'mae': result['mae'],
                'time_budget_minutes': result['time_budget_minutes'],
                'notes': f"AutoTabPFNRegressor ensemble, R²={result['r2']:.4f}"
            }])], ignore_index=True)
        
        comparison_df.to_csv('tabpfn_ensemble_comparison.csv', index=False)
        print(f"\n💾 Complete comparison saved to: tabpfn_ensemble_comparison.csv")
        
    else:
        print(f"\n❌ No successful ensemble results")
        print(f"This might be due to:")
        print(f"   • Insufficient memory for multiple TabPFN models")
        print(f"   • CPU-only computation being too slow")
        print(f"   • TabPFN Extensions compatibility issues")
        print(f"   • Time budget being too short for meaningful ensembles")
    
    print(f"\n🚀 NEXT STEPS:")
    if len(successful_results) > 0 and successful_results[0]['mae'] < 55.96:
        print(f"   1. 🎉 We achieved a new record! Document the breakthrough")
        print(f"   2. 🔬 Analyze what made this ensemble superior")
        print(f"   3. 🚀 Try even longer training times for further improvements")
        print(f"   4. 📊 Test ensemble on private cases")
    else:
        print(f"   1. 🖥️  Try running with GPU acceleration if available")
        print(f"   2. ⏱️  Increase time budgets for better ensemble diversity")
        print(f"   3. 🔧 Experiment with different ensemble configurations")
        print(f"   4. 💾 Consider the single TabPFN approach remains best for now")

if __name__ == "__main__":
    main() 