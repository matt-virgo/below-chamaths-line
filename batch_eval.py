#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture and features from ultra_deep_learning
from ultra_deep_learning import AttentionNet, create_ultra_features

def load_best_model():
    """Load the best model and its scalers once"""
    
    print("🔧 Loading best model and scalers...")
    
    # Best model details
    model_id = "QuantileTransformer_AttentionNet_WD2e-3"
    model_file = f"{model_id}_best.pth"
    scalers_file = f"{model_id}_scalers.pkl"
    
    # Load scalers
    try:
        with open(scalers_file, 'rb') as f:
            scalers = pickle.load(f)
            scaler_X = scalers['scaler_X']
            scaler_y = scalers['scaler_y']
        print(f"✅ Loaded scalers: {scalers_file}")
    except FileNotFoundError:
        # Fallback to best_overall files
        try:
            with open('best_overall_scalers.pkl', 'rb') as f:
                scalers = pickle.load(f)
                scaler_X = scalers['scaler_X']
                scaler_y = scalers['scaler_y']
            model_file = 'best_overall_model.pth'
            print(f"✅ Loaded fallback scalers: best_overall_scalers.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("No model scalers found!")
    
    # Create model with correct architecture (58 input features)
    model = AttentionNet(input_size=58, hidden_size=256)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        print(f"✅ Loaded model: {model_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {model_file} not found!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model has {total_params:,} parameters")
    
    return model, scaler_X, scaler_y

def batch_evaluate():
    """
    Evaluate our implementation against test cases (like eval.sh but much faster)
    """
    
    print("🧾 Black Box Challenge - Reimbursement System Evaluation (Fast Python)")
    print("=" * 75)
    print()
    
    # Check if test cases exist - try multiple possible files
    test_files = ['public_cases.json', 'test_cases.json']
    cases_file = None
    
    for file in test_files:
        try:
            with open(file, 'r') as f:
                cases = json.load(f)
            cases_file = file
            print(f"📂 Using test cases from: {cases_file}")
            print(f"✅ Loaded {len(cases)} test cases")
            break
        except FileNotFoundError:
            continue
    
    if cases_file is None:
        print("❌ Error: No test cases file found!")
        print("   Looked for: public_cases.json, test_cases.json")
        return False
    
    # Load model and scalers once
    model, scaler_X, scaler_y = load_best_model()
    
    print(f"📊 Running evaluation against {len(cases)} test cases...")
    print()
    
    start_time = time.time()
    
    # Prepare data for batch processing
    input_data = []
    expected_outputs = []
    case_details = []
    
    for i, case in enumerate(cases):
        # Handle both formats: nested "input" (test_cases.json) or flat (public_cases.json)
        if 'input' in case:
            # Nested format (test_cases.json)
            trip_duration = case['input']['trip_duration_days']
            miles_traveled = case['input']['miles_traveled'] 
            receipts_amount = case['input']['total_receipts_amount']
            expected_output = case['expected_output']
        else:
            # Flat format with expected_output (public_cases.json)
            trip_duration = case['trip_duration_days']
            miles_traveled = case['miles_traveled']
            receipts_amount = case['total_receipts_amount']
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
    
    # Create features for all cases at once
    print(f"⚙️ Engineering features...")
    features_df = create_ultra_features(input_df)
    print(f"✅ Created {features_df.shape[1]} features for all cases")
    
    # Scale features
    print(f"📏 Scaling features...")
    X_scaled = scaler_X.transform(features_df)
    
    # Make batch predictions
    print(f"🔮 Making batch predictions...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_scaled_pred = model(X_tensor)
        
        # Reverse scaling to get actual dollar amounts
        predictions = scaler_y.inverse_transform(y_scaled_pred.numpy().reshape(-1, 1)).flatten()
    
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
    print(f"🚀 Speedup vs eval.sh: ~{(1.5 * len(cases)) / total_time:.0f}x faster")
    print()
    
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(cases)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print()
    
    # Calculate score (like eval.sh)
    score = avg_error * 100 + (len(cases) - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print()
    
    # Provide feedback based on exact matches
    if exact_matches == len(cases):
        print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! You are very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! You have captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! You understand some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")
    
    print()
    print("💡 Tips for improvement:")
    if exact_matches < len(cases):
        print("  Check these high-error cases:")
        
        # Find top 5 highest error cases
        high_error_indices = np.argsort(errors)[-5:][::-1]  # Top 5 in descending order
        
        for i, idx in enumerate(high_error_indices):
            case = case_details[idx]
            expected = expected_outputs[idx]
            actual = predictions[idx]
            error = errors[idx]
            print(f"    Case {case['case_id']}: {case['trip_duration_days']} days, "
                  f"{case['miles_traveled']} miles, ${case['total_receipts_amount']:.2f} receipts")
            print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.2f}")
    
    # Show best predictions too
    print()
    print("🏆 Best predictions (lowest errors):")
    best_indices = np.argsort(errors)[:5]  # Top 5 lowest errors
    
    for i, idx in enumerate(best_indices):
        case = case_details[idx]
        expected = expected_outputs[idx]
        actual = predictions[idx]
        error = errors[idx]
        print(f"    Case {case['case_id']}: {case['trip_duration_days']} days, "
              f"{case['miles_traveled']} miles, ${case['total_receipts_amount']:.2f} receipts")
        print(f"      Expected: ${expected:.2f}, Got: ${actual:.2f}, Error: ${error:.4f}")
    
    return True

def main():
    """Main function"""
    
    try:
        success = batch_evaluate()
        if success:
            print(f"\n🎉 Evaluation completed successfully!")
        else:
            print(f"❌ Evaluation failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 