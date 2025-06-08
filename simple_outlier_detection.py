#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

def detect_statistical_outliers(df, features, contamination=0.1):
    """Detect outliers using multiple statistical methods"""
    X = df[features].values
    
    # Z-score method
    z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
    z_outliers = (z_scores > 3).any(axis=1)
    
    # IQR method
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    iqr_outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    iso_outliers = iso_forest.fit_predict(X) == -1
    
    # Combine methods
    consensus_outliers = z_outliers | iqr_outliers | iso_outliers
    
    return {
        'z_score_outliers': z_outliers,
        'iqr_outliers': iqr_outliers,
        'isolation_forest_outliers': iso_outliers,
        'consensus_outliers': consensus_outliers,
        'outlier_scores': iso_forest.decision_function(X)
    }

def detect_tabpfn_outliers(train_df, test_df, features):
    """Use TabPFN unsupervised model for outlier detection"""
    try:
        from tabpfn_extensions import TabPFNUnsupervisedModel
        
        print("   🎯 Using TabPFN Unsupervised Model...")
        
        # Prepare data
        X_train = train_df[features].values
        X_test = test_df[features].values
        X_combined = np.vstack([X_train, X_test])
        
        # Create unsupervised model
        unsupervised_model = TabPFNUnsupervisedModel(device='cpu')
        print("   📱 Using CPU mode for compatibility")
        
        # Fit on training data
        print(f"   🏋️ Training on {len(X_train)} samples...")
        unsupervised_model.fit(X_train)
        
        # Get outlier scores for all data
        print("   🔮 Computing outlier scores...")
        outlier_scores = unsupervised_model.score_samples(X_combined)
        
        # Determine outlier threshold (bottom 10%)
        threshold = np.percentile(outlier_scores, 10)
        tabpfn_outliers = outlier_scores < threshold
        
        return {
            'tabpfn_outliers': tabpfn_outliers,
            'tabpfn_scores': outlier_scores,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ TabPFN Unsupervised failed: {str(e)}")
        return {'success': False, 'error': str(e)}

def analyze_prediction_errors(df, model_predictions):
    """Analyze which samples have high prediction errors"""
    if 'tabpfn_prediction' in df.columns:
        errors = np.abs(df['reimbursement'] - df['tabpfn_prediction'])
        error_threshold = np.percentile(errors, 90)  # Top 10% errors
        error_outliers = errors > error_threshold
        return {
            'error_outliers': error_outliers,
            'prediction_errors': errors
        }
    return None

def main():
    print("🔍 Comprehensive Outlier Detection Analysis")
    print("="*70)
    print("Identifying problematic samples using multiple methods")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Combine datasets for analysis
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"Dataset analysis:")
    print(f"   📊 Train samples: {len(train_df)}")
    print(f"   📊 Test samples: {len(test_df)}")
    print(f"   📊 Total samples: {len(combined_df)}")
    
    # Raw features for outlier detection
    raw_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    
    print(f"\n{'='*70}")
    print(f"🔍 STATISTICAL OUTLIER DETECTION")
    print(f"{'='*70}")
    
    # Statistical outlier detection
    print("Running statistical outlier detection methods...")
    stat_results = detect_statistical_outliers(combined_df, raw_features)
    
    for method, outliers in stat_results.items():
        if method != 'outlier_scores':
            num_outliers = np.sum(outliers)
            percentage = (num_outliers / len(combined_df)) * 100
            print(f"   {method}: {num_outliers}/{len(combined_df)} ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"🔍 TabPFN UNSUPERVISED OUTLIER DETECTION")
    print(f"{'='*70}")
    
    # TabPFN unsupervised outlier detection
    tabpfn_results = detect_tabpfn_outliers(train_df, test_df, raw_features)
    
    if tabpfn_results['success']:
        tabpfn_outliers = tabpfn_results['tabpfn_outliers']
        num_tabpfn_outliers = np.sum(tabpfn_outliers)
        percentage = (num_tabpfn_outliers / len(combined_df)) * 100
        print(f"   TabPFN Unsupervised: {num_tabpfn_outliers}/{len(combined_df)} ({percentage:.1f}%)")
        
        combined_df['tabpfn_outlier_score'] = tabpfn_results['tabpfn_scores']
        combined_df['tabpfn_is_outlier'] = tabpfn_outliers
    
    # Add all outlier indicators to combined_df
    for method, outliers in stat_results.items():
        if method != 'outlier_scores':
            combined_df[f'{method}'] = outliers
    
    combined_df['isolation_forest_score'] = stat_results['outlier_scores']
    
    print(f"\n{'='*70}")
    print(f"🔍 PREDICTION ERROR ANALYSIS")
    print(f"{'='*70}")
    
    # Load TabPFN predictions if available
    try:
        tabpfn_results_df = pd.read_csv('tabpfn_v1_engineered_results.csv')
        if len(tabpfn_results_df) == len(test_df):
            # Add prediction errors for test set
            test_errors = np.abs(test_df['reimbursement'] - tabpfn_results_df['tabpfn_prediction'])
            error_threshold = np.percentile(test_errors, 90)
            
            # Create full error array
            full_errors = np.full(len(combined_df), np.nan)
            test_indices = combined_df['dataset'] == 'test'
            full_errors[test_indices] = test_errors
            
            combined_df['prediction_error'] = full_errors
            combined_df['high_error_outlier'] = full_errors > error_threshold
            
            print(f"   High prediction error outliers: {np.sum(~np.isnan(full_errors) & (full_errors > error_threshold))}")
            print(f"   Error threshold (90th percentile): ${error_threshold:.2f}")
        
    except FileNotFoundError:
        print("   📝 TabPFN prediction results not found")
    
    print(f"\n{'='*70}")
    print(f"📊 OUTLIER CONSENSUS ANALYSIS")
    print(f"{'='*70}")
    
    # Find consensus outliers (detected by multiple methods)
    outlier_methods = ['z_score_outliers', 'iqr_outliers', 'isolation_forest_outliers']
    if 'tabpfn_is_outlier' in combined_df.columns:
        outlier_methods.append('tabpfn_is_outlier')
    
    # Count how many methods detected each sample as outlier
    outlier_counts = combined_df[outlier_methods].sum(axis=1)
    combined_df['outlier_consensus_count'] = outlier_counts
    
    # Consensus outliers (detected by at least 2 methods)
    consensus_outliers = outlier_counts >= 2
    combined_df['consensus_outlier'] = consensus_outliers
    
    num_consensus = np.sum(consensus_outliers)
    print(f"Consensus outliers (2+ methods): {num_consensus}/{len(combined_df)} ({num_consensus/len(combined_df)*100:.1f}%)")
    
    # Show method agreement
    print(f"\n📊 METHOD AGREEMENT:")
    for i in range(1, len(outlier_methods) + 1):
        count = np.sum(outlier_counts == i)
        if count > 0:
            print(f"   Detected by {i} method(s): {count} samples")
    
    # Analyze consensus outliers
    if num_consensus > 0:
        consensus_samples = combined_df[consensus_outliers]
        
        print(f"\n🚨 TOP CONSENSUS OUTLIERS:")
        consensus_sorted = consensus_samples.sort_values('outlier_consensus_count', ascending=False)
        
        for i, (idx, row) in enumerate(consensus_sorted.head(15).iterrows()):
            dataset_type = row['dataset']
            original_idx = row['index']
            methods_count = row['outlier_consensus_count']
            print(f"   {i+1:2d}. {dataset_type.upper()}[{original_idx:3d}] | "
                  f"Days: {row['trip_duration_days']:2d} | "
                  f"Miles: {row['miles_traveled']:6.1f} | "
                  f"Receipts: ${row['total_receipts_amount']:7.2f} | "
                  f"Reimbursement: ${row['reimbursement']:8.2f} | "
                  f"Methods: {methods_count}/{len(outlier_methods)}")
        
        # Statistical comparison
        print(f"\n📈 CONSENSUS OUTLIERS vs NORMAL SAMPLES:")
        normal_samples = combined_df[~consensus_outliers]
        
        for feature in raw_features + ['reimbursement']:
            outlier_mean = consensus_samples[feature].mean()
            normal_mean = normal_samples[feature].mean()
            outlier_std = consensus_samples[feature].std()
            normal_std = normal_samples[feature].std()
            
            print(f"   {feature}:")
            print(f"      Outliers: μ={outlier_mean:.2f}, σ={outlier_std:.2f}")
            print(f"      Normal:   μ={normal_mean:.2f}, σ={normal_std:.2f}")
    
    # Save comprehensive results
    outlier_analysis_results = combined_df.copy()
    outlier_analysis_results.to_csv('comprehensive_outlier_analysis.csv', index=False)
    print(f"\n💾 Complete outlier analysis saved to: comprehensive_outlier_analysis.csv")
    
    # Summary insights
    print(f"\n🧠 OUTLIER DETECTION INSIGHTS:")
    print(f"   • Multiple methods provide robust outlier identification")
    print(f"   • Consensus outliers are likely the most problematic samples")
    print(f"   • Statistical methods catch extreme feature values")
    if tabpfn_results['success']:
        print(f"   • TabPFN unsupervised captures complex pattern deviations")
    print(f"   • High prediction errors indicate model difficulty")
    
    print(f"\n🚀 RECOMMENDED ACTIONS:")
    print(f"   1. Review consensus outliers manually for data quality")
    print(f"   2. Consider removing outliers and retrain models")
    print(f"   3. Use outlier-robust loss functions")
    print(f"   4. Weight samples inversely to outlier scores")
    print(f"   5. Investigate if outliers represent edge cases or errors")
    
    # Create outlier-free dataset suggestion
    if num_consensus > 0:
        clean_train = train_df[~consensus_outliers[:len(train_df)]]
        clean_test = test_df[~consensus_outliers[len(train_df):]]
        
        print(f"\n🧹 CLEAN DATASET SUGGESTION:")
        print(f"   Original: {len(train_df)} train + {len(test_df)} test")
        print(f"   Clean:    {len(clean_train)} train + {len(clean_test)} test")
        print(f"   Removed:  {len(train_df) - len(clean_train)} train + {len(test_df) - len(clean_test)} test outliers")

if __name__ == "__main__":
    main() 