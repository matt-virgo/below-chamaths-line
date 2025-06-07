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

def batch_process_cases(cases_file, output_file=None, results_txt_file=None):
    """
    Batch process all cases from a JSON file
    
    Args:
        cases_file (str): Path to JSON file with test cases
        output_file (str): Optional path to save detailed results CSV
        results_txt_file (str): Optional path to save simple results text file (one result per line)
    """
    
    print(f"📂 Loading cases from: {cases_file}")
    
    # Load test cases
    try:
        with open(cases_file, 'r') as f:
            cases = json.load(f)
        print(f"✅ Loaded {len(cases)} test cases")
    except FileNotFoundError:
        print(f"❌ Error: {cases_file} not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None
    
    # Load model and scalers once
    model, scaler_X, scaler_y = load_best_model()
    
    print(f"🚀 Starting batch processing...")
    start_time = time.time()
    
    # Prepare data for batch processing
    input_data = []
    case_info = []
    
    for i, case in enumerate(cases):
        # Handle both formats: nested "input" (test_cases.json) or flat (private_cases.json)
        if 'input' in case:
            # Nested format (test_cases.json)
            trip_duration = case['input']['trip_duration_days']
            miles_traveled = case['input']['miles_traveled'] 
            receipts_amount = case['input']['total_receipts_amount']
            expected_output = case.get('expected_output', None)
        else:
            # Flat format (private_cases.json)
            trip_duration = case['trip_duration_days']
            miles_traveled = case['miles_traveled']
            receipts_amount = case['total_receipts_amount']
            expected_output = case.get('expected_output', None)  # May not be present
        
        input_data.append({
            'trip_duration_days': trip_duration,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': receipts_amount
        })
        case_info.append({
            'case_id': i + 1,
            'trip_duration_days': trip_duration,
            'miles_traveled': miles_traveled,
            'total_receipts_amount': receipts_amount,
            'expected_output': expected_output
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
        y_pred = scaler_y.inverse_transform(y_scaled_pred.numpy().reshape(-1, 1)).flatten()
    
    # Compile results
    results = []
    for i, (case, prediction) in enumerate(zip(case_info, y_pred)):
        result = {
            'case_id': case['case_id'],
            'trip_duration_days': case['trip_duration_days'],
            'miles_traveled': case['miles_traveled'],
            'total_receipts_amount': case['total_receipts_amount'],
            'predicted_reimbursement': round(prediction, 2)
        }
        
        # Add expected output and error if available
        if case['expected_output'] is not None:
            result['expected_output'] = case['expected_output']
            result['error'] = round(case['expected_output'] - prediction, 2)
            result['abs_error'] = round(abs(case['expected_output'] - prediction), 2)
        
        results.append(result)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(cases)
    
    print(f"✅ Batch processing complete!")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"⏱️ Average per case: {avg_time:.4f} seconds ({1/avg_time:.1f} cases/sec)")
    print(f"🚀 Speedup vs individual runs: {1.5/avg_time:.1f}x faster")
    
    # Calculate stats if expected outputs are available
    has_expected = any(r.get('expected_output') is not None for r in results)
    if has_expected:
        errors = [r['abs_error'] for r in results if r.get('abs_error') is not None]
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean([e**2 for e in errors]))
        exact_matches = sum(1 for e in errors if e < 0.01)
        close_matches_1 = sum(1 for e in errors if e < 1.0)
        close_matches_5 = sum(1 for e in errors if e < 5.0)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   MAE: ${mae:.2f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   Exact matches (±$0.01): {exact_matches}/{len(errors)} ({exact_matches/len(errors)*100:.1f}%)")
        print(f"   Close matches (±$1.00): {close_matches_1}/{len(errors)} ({close_matches_1/len(errors)*100:.1f}%)")
        print(f"   Close matches (±$5.00): {close_matches_5}/{len(errors)} ({close_matches_5/len(errors)*100:.1f}%)")
    else:
        print(f"\n📊 No expected outputs available (private cases)")
    
    # Save detailed results to CSV if requested
    if output_file:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        print(f"💾 Detailed results saved to: {output_file}")
    
    # Save simple results text file (matching generate_results.sh format)
    if results_txt_file:
        with open(results_txt_file, 'w') as f:
            for result in results:
                f.write(f"{result['predicted_reimbursement']:.2f}\n")
        print(f"📝 Simple results saved to: {results_txt_file}")
        print(f"   📋 Format: One result per line, matching case order")
        print(f"   📊 Line 1 = Result for case 1, Line 2 = Result for case 2, etc.")
    
    # Show first few predictions
    print(f"\n🔍 Sample Predictions:")
    for i in range(min(10, len(results))):
        r = results[i]
        expected_str = f", Expected: ${r['expected_output']:.2f}" if r.get('expected_output') else ""
        error_str = f", Error: ${r['abs_error']:.2f}" if r.get('abs_error') else ""
        print(f"   Case {r['case_id']}: {r['trip_duration_days']} days, {r['miles_traveled']} miles, "
              f"${r['total_receipts_amount']:.2f} → ${r['predicted_reimbursement']:.2f}{expected_str}{error_str}")
    
    return results

def main():
    """Main function"""
    
    # Default to private_cases.json
    cases_file = 'private_cases.json'
    output_file = 'private_cases_results.csv'
    results_txt_file = 'private_results.txt'  # Match generate_results.sh output
    
    # Check command line arguments
    if len(sys.argv) > 1:
        cases_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        results_txt_file = sys.argv[3]
    
    print(f"🎯 Batch Processing ACME Travel Reimbursement")
    print(f"=" * 60)
    print(f"📁 Input file: {cases_file}")
    print(f"💾 Detailed output: {output_file}")
    print(f"📝 Simple output: {results_txt_file}")
    print()
    
    try:
        results = batch_process_cases(cases_file, output_file, results_txt_file)
        
        if results:
            print(f"\n🎉 Successfully processed {len(results)} cases!")
            print(f"📋 Detailed results: {output_file}")
            print(f"📄 Simple results: {results_txt_file}")
            print(f"\n📈 File formats:")
            print(f"   CSV: Detailed with case_id, inputs, predictions, errors (if available)")
            print(f"   TXT: Simple format - one result per line (matches generate_results.sh)")
        else:
            print(f"❌ Processing failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 