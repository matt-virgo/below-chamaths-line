#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Import the feature engineering from the enhanced pricing psychology script
from tabpfn_enhanced_pricing_psychology import engineer_enhanced_features

def load_enhanced_model():
    """Load and initialize the enhanced TabPFN model"""
    
    print("🔧 Loading enhanced TabPFN model...")
    
    try:
        from tabpfn import TabPFNRegressor
        
        # Load training data to train the model
        print("📚 Loading training data...")
        with open('train_cases.json', 'r') as f:
            train_data = json.load(f)
        
        # Convert to DataFrame
        train_df = pd.DataFrame([
            {
                'trip_duration_days': case['input']['trip_duration_days'],
                'miles_traveled': case['input']['miles_traveled'],
                'total_receipts_amount': case['input']['total_receipts_amount'],
                'reimbursement': case['expected_output']
            }
            for case in train_data
        ])
        
        print(f"📊 Training data: {len(train_df)} samples")
        
        # Create enhanced features
        print("🏢 Engineering enhanced features for training...")
        X_train = engineer_enhanced_features(train_df)
        y_train = train_df['reimbursement'].values
        
        print(f"✅ Features created: {X_train.shape[1]} enhanced business features")
        
        # Initialize and train TabPFN
        print("🚀 Training TabPFN...")
        tabpfn = TabPFNRegressor(device='cpu')
        
        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        y_train_np = y_train.astype(np.float32)
        
        tabpfn.fit(X_train_np, y_train_np)
        
        print(f"✅ TabPFN trained successfully!")
        
        return tabpfn, X_train.shape[1]
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def batch_evaluate_enhanced():
    """
    Evaluate enhanced pricing psychology model against public cases
    """
    
    print("🧾 Enhanced Pricing Psychology - Public Cases Evaluation")
    print("=" * 75)
    print("💰 Using TabPFN with 12 enhanced pricing psychology features")
    print()
    
    # Load public cases
    try:
        with open('public_cases.json', 'r') as f:
            cases = json.load(f)
        print(f"📂 Loaded {len(cases)} public test cases")
    except FileNotFoundError:
        print("❌ Error: public_cases.json not found!")
        return False
    
    # Load enhanced model
    model, num_features = load_enhanced_model()
    if model is None:
        return False
    
    print(f"📊 Running evaluation against {len(cases)} public cases...")
    print(f"🏢 Using {num_features} enhanced business features")
    print()
    
    start_time = time.time()
    
    # Prepare data for batch processing
    input_data = []
    expected_outputs = []
    case_details = []
    
    for i, case in enumerate(cases):
        # Handle nested format
        trip_duration = case['input']['trip_duration_days']
        miles_traveled = case['input']['miles_traveled'] 
        receipts_amount = case['input']['total_receipts_amount']
        expected_output = case['expected_output']
        
        input_data.append({
            'trip_duration_days': trip_duration,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': receipts_amount
        })
        expected_outputs.append(expected_output)
        case_details.append({
            'case_id': i + 1,
            'trip_duration_days': trip_duration,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': receipts_amount
        })
    
    # Convert to DataFrame for feature engineering
    input_df = pd.DataFrame(input_data)
    print(f"📈 Created input DataFrame with {len(input_df)} cases")
    
    # Create enhanced features for all cases at once
    print(f"⚙️ Engineering enhanced features for public cases...")
    features_df = engineer_enhanced_features(input_df)
    print(f"✅ Created {features_df.shape[1]} enhanced features for all cases")
    
    # Make batch predictions
    print(f"🔮 Making batch predictions with TabPFN...")
    
    # Convert to numpy arrays
    X_test_np = features_df.values.astype(np.float32)
    predictions = model.predict(X_test_np)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate evaluation metrics
    expected_outputs = np.array(expected_outputs)
    predictions = np.array(predictions)
    
    # Count successful runs (all should be successful in batch mode)
    successful_runs = len(predictions)
    
    # Calculate errors
    errors = np.abs(expected_outputs - predictions)
    
    # Count precision matches
    exact_matches = np.sum(errors < 0.01)
    close_matches = np.sum(errors < 1.0)
    
    # Calculate statistics
    total_error = np.sum(errors)
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    max_error_idx = np.argmax(errors)
    
    # Calculate percentages
    exact_pct = (exact_matches / successful_runs) * 100
    close_pct = (close_matches / successful_runs) * 100
    
    print(f"✅ Evaluation Complete!")
    print(f"⏱️ Total evaluation time: {total_time:.2f} seconds")
    print(f"⏱️ Average per case: {total_time/len(cases):.4f} seconds ({len(cases)/total_time:.1f} cases/sec)")
    print()
    
    print("📈 Enhanced Pricing Psychology Results:")
    print(f"  Total test cases: {len(cases)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error (MAE): ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print()
    
    # Calculate score (like eval.sh)
    score = avg_error * 100 + (len(cases) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print()
    
    # Analyze pricing patterns in predictions
    print("💰 PRICING PATTERN ANALYSIS:")
    pricing_features = features_df[['is_receipt_49_or_99_cents', 'is_receipt_psychological_pricing', 
                                   'is_receipt_round_dollar', 'is_receipt_9_ending']]
    
    pricing_49_99_count = (pricing_features['is_receipt_49_or_99_cents'] == 1).sum()
    pricing_psych_count = (pricing_features['is_receipt_psychological_pricing'] == 1).sum()
    pricing_round_count = (pricing_features['is_receipt_round_dollar'] == 1).sum()
    pricing_9_ending_count = (pricing_features['is_receipt_9_ending'] == 1).sum()
    
    print(f"   🎯 Cases with .49/.99 cents: {pricing_49_99_count} ({pricing_49_99_count/len(cases)*100:.1f}%)")
    print(f"   💰 Cases with psychological pricing: {pricing_psych_count} ({pricing_psych_count/len(cases)*100:.1f}%)")
    print(f"   🔢 Cases with round dollars: {pricing_round_count} ({pricing_round_count/len(cases)*100:.1f}%)")
    print(f"   9️⃣ Cases ending in 9: {pricing_9_ending_count} ({pricing_9_ending_count/len(cases)*100:.1f}%)")
    
    # Analyze errors by pricing patterns
    if pricing_49_99_count > 0:
        pricing_49_99_mask = pricing_features['is_receipt_49_or_99_cents'] == 1
        pricing_errors = errors[pricing_49_99_mask]
        non_pricing_errors = errors[~pricing_49_99_mask]
        
        pricing_mae = np.mean(pricing_errors)
        non_pricing_mae = np.mean(non_pricing_errors)
        
        print(f"   📊 .49/.99 cases MAE: ${pricing_mae:.2f}")
        print(f"   📊 Non-.49/.99 cases MAE: ${non_pricing_mae:.2f}")
        
        if pricing_mae < non_pricing_mae:
            improvement = non_pricing_mae - pricing_mae
            print(f"   🎉 .49/.99 cases perform ${improvement:.2f} BETTER!")
        else:
            gap = pricing_mae - non_pricing_mae
            print(f"   📈 .49/.99 cases still ${gap:.2f} higher MAE")
    
    print()
    
    # Show top errors
    print("🔍 TOP 10 WORST PREDICTIONS:")
    top_errors_idx = np.argsort(errors)[-10:][::-1]
    
    for rank, idx in enumerate(top_errors_idx, 1):
        case_info = case_details[idx]
        actual = expected_outputs[idx]
        predicted = predictions[idx]
        error = errors[idx]
        
        has_49_99 = "💰" if pricing_features.iloc[idx]['is_receipt_49_or_99_cents'] == 1 else "  "
        
        print(f"  {rank:2d}. {has_49_99} Case {case_info['case_id']:4d}: "
              f"${actual:8.2f} → ${predicted:8.2f} (Error: ${error:6.2f}) "
              f"[{case_info['trip_duration_days']}d, {case_info['miles_traveled']}mi, ${case_info['total_receipts_amount']:.2f}]")
    
    print()
    
    # Save detailed results
    results_data = []
    for i in range(len(cases)):
        case_info = case_details[i]
        results_data.append({
            'case_id': case_info['case_id'],
            'trip_duration_days': case_info['trip_duration_days'],
            'miles_traveled': case_info['miles_traveled'],
            'total_receipts_amount': case_info['total_receipts_amount'],
            'expected_output': expected_outputs[i],
            'predicted_output': predictions[i],
            'absolute_error': errors[i],
            'has_49_99_cents': pricing_features.iloc[i]['is_receipt_49_or_99_cents'],
            'has_psychological_pricing': pricing_features.iloc[i]['is_receipt_psychological_pricing'],
            'has_round_dollar': pricing_features.iloc[i]['is_receipt_round_dollar'],
            'has_9_ending': pricing_features.iloc[i]['is_receipt_9_ending']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv('enhanced_pricing_public_results.csv', index=False)
    
    # Create public_cases_predictions.csv in the required format
    predictions_data = []
    for i in range(len(cases)):
        predictions_data.append({
            'trip_duration_days': case_details[i]['trip_duration_days'],
            'miles_traveled': case_details[i]['miles_traveled'],
            'total_receipts_amount': case_details[i]['total_receipts_amount'],
            'predicted_reimbursement': float(predictions[i])
        })
    
    predictions_df = pd.DataFrame(predictions_data)
    predictions_df.to_csv('public_cases_predictions.csv', index=False)
    
    print(f"💾 Results saved:")
    print(f"   📊 Detailed results: enhanced_pricing_public_results.csv")
    print(f"   🔮 Predictions: public_cases_predictions.csv")
    print()
    
    # Compare to previous best results
    print("📈 PERFORMANCE COMPARISON:")
    previous_results = [
        ("TabPFN Business Rules (Original)", 55.21),
        ("V1 Neural Networks", 57.35),
        ("TabPFN Enhanced Business Rules", avg_error)
    ]
    
    print(f"   🆕 Enhanced Pricing Psychology: ${avg_error:.2f} MAE")
    print(f"   📊 vs Original TabPFN Business Rules: ${(avg_error - 55.21):+.2f}")
    
    if avg_error < 55.21:
        improvement = 55.21 - avg_error
        improvement_pct = (improvement / 55.21) * 100
        print(f"   🎉 NEW RECORD! Improved by ${improvement:.2f} ({improvement_pct:.2f}%)")
        print(f"   💰 Enhanced pricing psychology features WORKED!")
    elif avg_error < 56.0:
        print(f"   🎯 Very competitive performance!")
        print(f"   💰 Enhanced pricing features showing promise")
    else:
        print(f"   📊 Room for improvement in pricing psychology approach")
    
    return True

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Enhanced Pricing Psychology Batch Evaluator")
        print("Usage: python batch_eval_enhanced_pricing.py")
        print()
        print("Evaluates the enhanced TabPFN model with 12 pricing psychology features")
        print("against public test cases and generates predictions.")
        return
    
    success = batch_evaluate_enhanced()
    
    if success:
        print("🎉 Enhanced pricing psychology evaluation completed successfully!")
    else:
        print("❌ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 