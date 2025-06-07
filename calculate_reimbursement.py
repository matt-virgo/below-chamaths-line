#!/usr/bin/env python3

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture and features from ultra_deep_learning
from ultra_deep_learning import AttentionNet, create_ultra_features

def load_best_model():
    """Load the best model and its scalers"""
    
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
    except FileNotFoundError:
        # Fallback to best_overall files
        try:
            with open('best_overall_scalers.pkl', 'rb') as f:
                scalers = pickle.load(f)
                scaler_X = scalers['scaler_X']
                scaler_y = scalers['scaler_y']
            model_file = 'best_overall_model.pth'
        except FileNotFoundError:
            raise FileNotFoundError("No model scalers found!")
    
    # Create model with correct architecture (58 input features)
    model = AttentionNet(input_size=58, hidden_size=256)
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {model_file} not found!")
    
    return model, scaler_X, scaler_y

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using our best trained model
    
    Args:
        trip_duration_days (float): Number of days for the trip
        miles_traveled (float): Total miles traveled
        total_receipts_amount (float): Total amount from receipts
    
    Returns:
        float: Calculated reimbursement amount
    """
    
    # Create input DataFrame with the required format
    input_df = pd.DataFrame([{
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }])
    
    # Create features using the same feature engineering
    features_df = create_ultra_features(input_df)
    
    # Load model and scalers
    model, scaler_X, scaler_y = load_best_model()
    
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

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        # Parse command line arguments
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        # Validate inputs
        if trip_duration_days <= 0:
            raise ValueError("Trip duration must be positive")
        if miles_traveled < 0:
            raise ValueError("Miles traveled must be non-negative")
        if total_receipts_amount < 0:
            raise ValueError("Total receipts amount must be non-negative")
        
        # Calculate reimbursement
        reimbursement = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        
        # Output the result (single number as required)
        print(f"{reimbursement:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure the model files are present in the current directory.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 