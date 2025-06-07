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

def predict_single_case(trip_duration_days, miles_traveled, total_receipts_amount, model, scaler_X, scaler_y):
    """
    Make a prediction for a single case (exactly like calculate_reimbursement.py)
    """
    
    # Create input DataFrame with the required format
    input_df = pd.DataFrame([{
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }])
    
    # Create features using the same feature engineering
    features_df = create_ultra_features(input_df)
    
    # Scale features
    X_scaled = scaler_X.transform(features_df)
    
    # Make prediction
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_scaled_pred = model(X_tensor)
        
        # Reverse scaling to get actual dollar amount
        y_pred = scaler_y.inverse_transform(y_scaled_pred.numpy().reshape(-1, 1))
        reimbursement = y_pred[0][0]
    
    return reimbursement

def generate_private_results():
    """
    Generate private_results.txt exactly like generate_results.sh but much faster
    """
    
    print("🧾 Black Box Challenge - Generating Private Results (Python)")
    print("=" * 65)
    print()
    
    # Check if private cases exist
    cases_file = 'private_cases.json'
    results_file = 'private_results.txt'
    
    print(f"📂 Loading cases from: {cases_file}")
    
    try:
        with open(cases_file, 'r') as f:
            cases = json.load(f)
        print(f"✅ Loaded {len(cases)} test cases")
    except FileNotFoundError:
        print(f"❌ Error: {cases_file} not found!")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return False
    
    # Load model and scalers once (much faster than loading per case)
    model, scaler_X, scaler_y = load_best_model()
    
    print(f"📊 Processing {len(cases)} test cases and generating results...")
    print(f"📝 Output will be saved to {results_file}")
    print()
    
    start_time = time.time()
    
    # Process each case individually (like generate_results.sh but in Python)
    results = []
    
    with open(results_file, 'w') as f:
        for i, case in enumerate(cases):
            if i % 1000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(cases) - i) / rate
                print(f"Progress: {i}/{len(cases)} cases processed ({rate:.1f} cases/sec, ETA: {eta:.1f}s)...")
            
            # Extract case data (handles flat format from private_cases.json)
            trip_duration = case['trip_duration_days']
            miles_traveled = case['miles_traveled']
            receipts_amount = case['total_receipts_amount']
            
            try:
                # Make prediction for this single case
                prediction = predict_single_case(trip_duration, miles_traveled, receipts_amount, 
                                               model, scaler_X, scaler_y)
                
                # Format to 2 decimal places (matching our run.sh output)
                formatted_result = f"{prediction:.2f}"
                f.write(formatted_result + "\n")
                results.append(prediction)
                
            except Exception as e:
                print(f"Error on case {i+1}: {e}")
                f.write("ERROR\n")
                results.append(None)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(cases)
    
    print(f"\n✅ Results generated successfully!")
    print(f"📄 Output saved to {results_file}")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    print(f"⏱️ Average per case: {avg_time:.4f} seconds ({1/avg_time:.1f} cases/sec)")
    print(f"🚀 Speedup vs generate_results.sh: ~{1.5/avg_time:.0f}x faster")
    print(f"📊 Each line contains the result for the corresponding test case in {cases_file}")
    
    print(f"\n🎯 File format:")
    print(f"  Line 1: Result for private_cases.json[0]")
    print(f"  Line 2: Result for private_cases.json[1]")
    print(f"  Line 3: Result for private_cases.json[2]")
    print(f"  ...")
    print(f"  Line {len(cases)}: Result for private_cases.json[{len(cases)-1}]")
    
    # Show sample results
    print(f"\n🔍 Sample Results:")
    for i in range(min(5, len(cases))):
        case = cases[i]
        result = results[i] if results[i] is not None else "ERROR"
        print(f"   Case {i+1}: {case['trip_duration_days']} days, {case['miles_traveled']} miles, "
              f"${case['total_receipts_amount']:.2f} → {result}")
    
    return True

def main():
    """Main function"""
    
    try:
        success = generate_private_results()
        if success:
            print(f"\n🎉 Private results generated successfully!")
            print(f"📋 Ready for submission: private_results.txt")
        else:
            print(f"❌ Failed to generate results!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 