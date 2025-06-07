#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture from mega_ultra_resnet
from mega_ultra_resnet import MegaUltraResNet, create_mega_features, load_data

def evaluate_current_checkpoint():
    """Load and evaluate the current best checkpoint"""
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating mega-comprehensive features...")
    X_train = create_mega_features(train_df)
    X_test = create_mega_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Use same scaling as in training
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create test data loader
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled), 
        torch.FloatTensor(y_test_scaled)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model with same architecture
    input_size = X_train_scaled.shape[1]
    model = MegaUltraResNet(
        input_size=input_size, 
        hidden_size=512,
        num_blocks=16,
        dropout_rate=0.08
    )
    
    # Load the current best checkpoint
    try:
        print("\nLoading current best checkpoint...")
        model.load_state_dict(torch.load('mega_best_model.pth', map_location='cpu'))
        model.eval()
        print("✅ Successfully loaded checkpoint!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
    except FileNotFoundError:
        print("❌ No checkpoint found yet!")
        return
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Evaluate on test set
    print("\n=== CURRENT CHECKPOINT EVALUATION ===")
    device = torch.device('cpu')  # Use CPU for quick evaluation
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
    
    # Reverse scaling
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    # Calculate precision metrics
    exact_matches = np.sum(np.abs(actuals - predictions) < 0.01)
    close_matches_1 = np.sum(np.abs(actuals - predictions) < 1.0)
    close_matches_5 = np.sum(np.abs(actuals - predictions) < 5.0)
    close_matches_10 = np.sum(np.abs(actuals - predictions) < 10.0)
    
    print(f"🎯 Current Performance:")
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
    
    # Compare to previous best
    print(f"\n📊 Performance Comparison:")
    print(f"   Previous best (UltraResNet): $58.91 MAE")
    print(f"   Current checkpoint: ${mae:.2f} MAE")
    if mae < 58.91:
        improvement = 58.91 - mae
        print(f"   🚀 IMPROVEMENT: ${improvement:.2f} better!")
    else:
        difference = mae - 58.91
        print(f"   Still need: ${difference:.2f} improvement")
    
    return mae, r2, exact_matches

if __name__ == "__main__":
    evaluate_current_checkpoint() 