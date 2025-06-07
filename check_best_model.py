#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture and features from ultra_deep_learning
from ultra_deep_learning import UltraResNet, create_ultra_features, load_data

def evaluate_best_model():
    """Load and evaluate the best_model.pth checkpoint (UltraResNet that achieved $58.91 MAE)"""
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra features...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Use RobustScaler (same as in the original training)
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Create test data loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create UltraResNet model with same architecture as in ultra_deep_learning.py
    input_size = X_train_scaled.shape[1]
    model = UltraResNet(input_size=input_size, hidden_size=256, num_blocks=8)
    
    # Load the best model checkpoint
    try:
        print("\nLoading best_model.pth checkpoint...")
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        model.eval()
        print("✅ Successfully loaded best_model.pth!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
    except FileNotFoundError:
        print("❌ best_model.pth not found!")
        return
    except Exception as e:
        print(f"❌ Error loading best_model.pth: {e}")
        return
    
    # Evaluate on test set
    print("\n=== BEST MODEL CHECKPOINT EVALUATION ===")
    device = torch.device('cpu')
    model.to(device)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    # Calculate precision metrics
    exact_matches = np.sum(np.abs(actuals - predictions) < 0.01)
    close_matches_1 = np.sum(np.abs(actuals - predictions) < 1.0)
    close_matches_5 = np.sum(np.abs(actuals - predictions) < 5.0)
    close_matches_10 = np.sum(np.abs(actuals - predictions) < 10.0)
    
    print(f"🎯 Original UltraResNet Performance:")
    print(f"   Test MAE: ${mae:.2f}")
    print(f"   Test RMSE: ${rmse:.2f}")
    print(f"   Test R²: {r2:.6f}")
    print(f"   Exact matches (±$0.01): {exact_matches}/{len(actuals)} ({exact_matches/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$1.00): {close_matches_1}/{len(actuals)} ({close_matches_1/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$5.00): {close_matches_5}/{len(actuals)} ({close_matches_5/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$10.00): {close_matches_10}/{len(actuals)} ({close_matches_10/len(actuals)*100:.1f}%)")
    
    # Show best predictions
    errors = np.abs(actuals - predictions)
    best_indices = np.argsort(errors)[:10]
    
    print(f"\n🏆 Top 10 most accurate predictions:")
    for i, idx in enumerate(best_indices):
        print(f"{i+1:2d}. Days: {test_df.iloc[idx]['trip_duration_days']:2.0f}, "
              f"Miles: {test_df.iloc[idx]['miles_traveled']:4.0f}, "
              f"Receipts: ${test_df.iloc[idx]['total_receipts_amount']:7.2f}, "
              f"Actual: ${actuals[idx]:7.2f}, "
              f"Predicted: ${predictions[idx]:7.2f}, "
              f"Error: ${errors[idx]:.4f}")
    
    # Verify this is the $58.91 model
    if abs(mae - 58.91) < 0.01:
        print(f"\n✅ CONFIRMED: This is the original UltraResNet that achieved $58.91 MAE!")
    else:
        print(f"\n⚠️  Note: Expected $58.91 MAE, got ${mae:.2f} MAE")
    
    print(f"\n💡 Model Details:")
    print(f"   Architecture: UltraResNet with 8 residual blocks")
    print(f"   Hidden size: 256")
    print(f"   Features: 58 (ultra-comprehensive feature set)")
    print(f"   Scaling: RobustScaler")
    print(f"   Parameters: {total_params:,}")
    
    return mae, r2, exact_matches

if __name__ == "__main__":
    evaluate_best_model() 