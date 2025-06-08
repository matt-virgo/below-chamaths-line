#!/usr/bin/env python3

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
            'reimbursement': case['expected_output'],
            'index': i,
            'dataset': 'train'
        }
        for i, case in enumerate(train_data)
    ])
    
    test_df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output'],
            'index': i,
            'dataset': 'test'
        }
        for i, case in enumerate(test_data)
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
    
    # Get feature columns (exclude metadata)
    feature_cols = [col for col in features_df.columns if col not in ['reimbursement', 'index', 'dataset']]
    
    return features_df[feature_cols]

def main():
    print("🔍 TabPFN Extensions - Outlier Detection Analysis")
    print("="*70)
    print("Identifying problematic samples in our travel reimbursement dataset")
    print()
    
    # First, try to install and import TabPFN Extensions
    try:
        from tabpfn_extensions.unsupervised.outlier_detection import OutlierDetector
        print("✅ TabPFN Extensions successfully imported!")
    except ImportError:
        print("❌ TabPFN Extensions not found. Installing...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git"])
            from tabpfn_extensions.unsupervised.outlier_detection import OutlierDetector
            print("✅ TabPFN Extensions installed and imported successfully!")
        except Exception as e:
            print(f"❌ Failed to install TabPFN Extensions: {str(e)}")
            print("🔧 Manual installation: pip install 'tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git'")
            return
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Combine datasets for comprehensive analysis
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Dataset analysis:")
    print(f"   📊 Train samples: {len(train_df)}")
    print(f"   📊 Test samples: {len(test_df)}")
    print(f"   📊 Total samples: {len(combined_df)}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Raw_Features_Outliers',
            'description': 'Outlier detection on 3 raw input features',
            'features': ['trip_duration_days', 'miles_traveled', 'total_receipts_amount'],
            'use_v1_features': False
        },
        {
            'name': 'V1_Engineered_Outliers',
            'description': 'Outlier detection on V1\'s 58 engineered features',
            'features': None,
            'use_v1_features': True
        }
    ]
    
    all_outlier_results = []
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"🔍 Analyzing: {config['name']}")
        print(f"📝 {config['description']}")
        print(f"{'='*70}")
        
        # Prepare features
        if config['use_v1_features']:
            print("Creating V1's comprehensive engineered features...")
            X_features = create_v1_ultra_features(combined_df)
            print(f"✨ Using {X_features.shape[1]} V1 engineered features")
        else:
            X_features = combined_df[config['features']]
            print(f"✨ Using {len(config['features'])} raw features:")
            for i, feature in enumerate(config['features'], 1):
                print(f"   {i}. {feature}")
        
        try:
            print(f"\n🎯 Running TabPFN outlier detection...")
            
            # Create outlier detector
            outlier_detector = OutlierDetector(device='cpu')
            print("   📱 Using CPU mode for compatibility")
            
            # Fit on training data only
            train_indices = combined_df['dataset'] == 'train'
            X_train_features = X_features[train_indices]
            
            print(f"   🏋️ Training outlier detector on {len(X_train_features)} training samples...")
            outlier_detector.fit(X_train_features)
            
            # Detect outliers on all data
            print("   🔮 Detecting outliers in all samples...")
            outlier_scores = outlier_detector.predict_scores(X_features)
            outlier_labels = outlier_detector.predict(X_features)
            
            # Analyze results
            num_outliers = np.sum(outlier_labels == -1)
            outlier_percentage = (num_outliers / len(combined_df)) * 100
            
            print(f"\n🎯 {config['name']} OUTLIER RESULTS:")
            print(f"   Total outliers detected: {num_outliers}/{len(combined_df)} ({outlier_percentage:.1f}%)")
            
            # Breakdown by dataset
            train_outliers = np.sum((outlier_labels == -1) & train_indices)
            test_outliers = np.sum((outlier_labels == -1) & (~train_indices))
            
            print(f"   Train outliers: {train_outliers}/{len(train_df)} ({train_outliers/len(train_df)*100:.1f}%)")
            print(f"   Test outliers: {test_outliers}/{len(test_df)} ({test_outliers/len(test_df)*100:.1f}%)")
            
            # Add outlier information to combined dataset
            combined_df[f'{config["name"]}_outlier_score'] = outlier_scores
            combined_df[f'{config["name"]}_is_outlier'] = (outlier_labels == -1)
            
            # Analyze outlier characteristics
            outlier_mask = outlier_labels == -1
            outlier_samples = combined_df[outlier_mask]
            normal_samples = combined_df[~outlier_mask]
            
            print(f"\n📊 OUTLIER CHARACTERISTICS:")
            print(f"   Outlier reimbursement stats:")
            print(f"      Mean: ${outlier_samples['reimbursement'].mean():.2f}")
            print(f"      Median: ${outlier_samples['reimbursement'].median():.2f}")
            print(f"      Std: ${outlier_samples['reimbursement'].std():.2f}")
            print(f"      Range: ${outlier_samples['reimbursement'].min():.2f} - ${outlier_samples['reimbursement'].max():.2f}")
            
            print(f"   Normal reimbursement stats:")
            print(f"      Mean: ${normal_samples['reimbursement'].mean():.2f}")
            print(f"      Median: ${normal_samples['reimbursement'].median():.2f}")
            print(f"      Std: ${normal_samples['reimbursement'].std():.2f}")
            print(f"      Range: ${normal_samples['reimbursement'].min():.2f} - ${normal_samples['reimbursement'].max():.2f}")
            
            # Show most extreme outliers
            if num_outliers > 0:
                outlier_samples_sorted = outlier_samples.sort_values(f'{config["name"]}_outlier_score')
                
                print(f"\n🚨 TOP 10 MOST EXTREME OUTLIERS:")
                for i, (idx, row) in enumerate(outlier_samples_sorted.head(10).iterrows()):
                    dataset_type = row['dataset']
                    original_idx = row['index']
                    score_col = f'{config["name"]}_outlier_score'
                    print(f"   {i+1:2d}. {dataset_type.upper()}[{original_idx:3d}] | "
                          f"Days: {row['trip_duration_days']:2d} | "
                          f"Miles: {row['miles_traveled']:6.1f} | "
                          f"Receipts: ${row['total_receipts_amount']:7.2f} | "
                          f"Reimbursement: ${row['reimbursement']:8.2f} | "
                          f"Score: {row[score_col]:.4f}")
            
            results = {
                'name': config['name'],
                'description': config['description'],
                'features_count': X_features.shape[1],
                'num_outliers': num_outliers,
                'outlier_percentage': outlier_percentage,
                'train_outliers': train_outliers,
                'test_outliers': test_outliers,
                'outlier_indices': combined_df[outlier_mask]['index'].tolist(),
                'outlier_datasets': combined_df[outlier_mask]['dataset'].tolist(),
                'success': True
            }
            
            all_outlier_results.append(results)
            
        except Exception as e:
            print(f"   ❌ Error with {config['name']}: {str(e)}")
            results = {
                'name': config['name'],
                'description': config['description'],
                'success': False,
                'error': str(e)
            }
            all_outlier_results.append(results)
    
    # Compare outlier detection methods
    print(f"\n{'='*80}")
    print(f"🔍 OUTLIER DETECTION COMPARISON:")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_outlier_results if r.get('success', False)]
    if len(successful_results) >= 2:
        raw_result = next((r for r in successful_results if 'Raw' in r['name']), None)
        eng_result = next((r for r in successful_results if 'V1' in r['name']), None)
        
        if raw_result and eng_result:
            print(f"📊 Raw Features Outliers:      {raw_result['num_outliers']:3d} ({raw_result['outlier_percentage']:4.1f}%)")
            print(f"📊 Engineered Features Outliers: {eng_result['num_outliers']:3d} ({eng_result['outlier_percentage']:4.1f}%)")
            
            # Find overlapping outliers
            raw_outlier_mask = combined_df.get('Raw_Features_Outliers_is_outlier', False)
            eng_outlier_mask = combined_df.get('V1_Engineered_Outliers_is_outlier', False)
            
            if 'Raw_Features_Outliers_is_outlier' in combined_df.columns and 'V1_Engineered_Outliers_is_outlier' in combined_df.columns:
                raw_outliers = set(zip(combined_df[combined_df['Raw_Features_Outliers_is_outlier']]['dataset'], 
                                      combined_df[combined_df['Raw_Features_Outliers_is_outlier']]['index']))
                eng_outliers = set(zip(combined_df[combined_df['V1_Engineered_Outliers_is_outlier']]['dataset'], 
                                      combined_df[combined_df['V1_Engineered_Outliers_is_outlier']]['index']))
            
            overlapping = raw_outliers.intersection(eng_outliers)
            raw_only = raw_outliers - eng_outliers
            eng_only = eng_outliers - raw_outliers
            
            print(f"\n🔬 OUTLIER OVERLAP ANALYSIS:")
            print(f"   Detected by both methods: {len(overlapping)}")
            print(f"   Raw features only: {len(raw_only)}")
            print(f"   Engineered features only: {len(eng_only)}")
            
            if len(overlapping) > 0:
                print(f"\n🎯 CONSENSUS OUTLIERS (detected by both methods):")
                for i, (dataset, idx) in enumerate(sorted(overlapping)):
                    row = combined_df[(combined_df['dataset'] == dataset) & (combined_df['index'] == idx)].iloc[0]
                    print(f"   {i+1:2d}. {dataset.upper()}[{idx:3d}] | "
                          f"Days: {row['trip_duration_days']:2d} | "
                          f"Miles: {row['miles_traveled']:6.1f} | "
                          f"Receipts: ${row['total_receipts_amount']:7.2f} | "
                          f"Reimbursement: ${row['reimbursement']:8.2f}")
    
    # Save comprehensive results
    outlier_analysis_results = combined_df.copy()
    outlier_analysis_results.to_csv('outlier_analysis_results.csv', index=False)
    print(f"\n💾 Complete outlier analysis saved to: outlier_analysis_results.csv")
    
    # Generate insights
    print(f"\n🧠 OUTLIER DETECTION INSIGHTS:")
    print(f"   • TabPFN's outlier detection uses its foundation model understanding")
    print(f"   • Outliers may represent edge cases or data quality issues")
    print(f"   • Different feature sets capture different types of anomalies")
    print(f"   • Consensus outliers are likely the most problematic samples")
    print(f"   • Removing outliers could potentially improve model performance")
    
    # Suggest next steps
    print(f"\n🚀 SUGGESTED NEXT STEPS:")
    print(f"   1. Manually review consensus outliers for data quality issues")
    print(f"   2. Train models with and without outliers to measure impact")
    print(f"   3. Use outlier scores for sample weighting during training")
    print(f"   4. Investigate if outliers follow specific patterns")
    print(f"   5. Consider robust loss functions for outlier-resistant training")

if __name__ == "__main__":
    main() 