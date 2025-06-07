#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the model architecture and features from ultra_deep_learning
from ultra_deep_learning import UltraResNet, create_ultra_features, load_data

def evaluate_robustscaler_ultraresnet():
    """Evaluate the RobustScaler_UltraResNet_WD1e-3 model that should achieve ~$58.91 MAE"""
    
    print("🎯 Evaluating RobustScaler UltraResNet Model")
    print("=" * 60)
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra features...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Load the saved scalers for this specific model
    model_id = "RobustScaler_UltraResNet_WD1e-3"
    scalers_file = f"{model_id}_scalers.pkl"
    
    print(f"Loading scalers: {scalers_file}")
    try:
        with open(scalers_file, 'rb') as f:
            scalers = pickle.load(f)
            scaler_X = scalers['scaler_X']
            scaler_y = scalers['scaler_y']
        print("✅ Scalers loaded successfully")
    except FileNotFoundError:
        print(f"❌ {scalers_file} not found!")
        return
    
    # Apply scaling
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create test data loader
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test_scaled))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Create UltraResNet model with same architecture
    input_size = X_train_scaled.shape[1]
    model = UltraResNet(input_size=input_size, hidden_size=256, num_blocks=8)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load the saved model
    model_file = f"{model_id}_best.pth"
    print(f"\nLoading model: {model_file}")
    try:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        print(f"✅ {model_file} loaded successfully!")
    except FileNotFoundError:
        print(f"❌ {model_file} not found!")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Evaluate on test set
    print(f"\n=== {model_id} EVALUATION ===")
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
    
    # Reverse scaling to get actual dollar amounts
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
    
    print(f"🎯 {model_id} Performance:")
    print(f"   Test MAE: ${mae:.2f}")
    print(f"   Test RMSE: ${rmse:.2f}")
    print(f"   Test R²: {r2:.6f}")
    print(f"   Exact matches (±$0.01): {exact_matches}/{len(actuals)} ({exact_matches/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$1.00): {close_matches_1}/{len(actuals)} ({close_matches_1/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$5.00): {close_matches_5}/{len(actuals)} ({close_matches_5/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$10.00): {close_matches_10}/{len(actuals)} ({close_matches_10/len(actuals)*100:.1f}%)")
    
    # Compare to target
    target_mae = 58.91
    print(f"\n📊 Performance Comparison:")
    print(f"   Target (original UltraResNet): ${target_mae:.2f} MAE")
    print(f"   Current {model_id}: ${mae:.2f} MAE")
    
    if mae < target_mae:
        improvement = target_mae - mae
        print(f"   🎯 NEW RECORD: ${improvement:.2f} better than original!")
    elif abs(mae - target_mae) < 1.0:
        difference = mae - target_mae
        print(f"   ✅ EXCELLENT: Only ${difference:.2f} away from target")
    elif abs(mae - target_mae) < 3.0:
        difference = mae - target_mae
        print(f"   ✅ VERY CLOSE: Only ${difference:.2f} away from target")
    else:
        difference = mae - target_mae
        print(f"   ⚠️  Gap: ${difference:.2f} away from target")
    
    # Check if this is the best performance
    if abs(mae - 58.91) < 0.1:
        print(f"\n🎯 CONFIRMED: This matches the original $58.91 UltraResNet performance!")
    elif mae < 58.91:
        print(f"\n🚀 AMAZING: This beats the original $58.91 performance!")
    
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
    
    print(f"\n💡 Model Details:")
    print(f"   Architecture: UltraResNet with 8 residual blocks")
    print(f"   Hidden size: 256")
    print(f"   Features: 58 (ultra-comprehensive feature set)")
    print(f"   Scaling: RobustScaler + StandardScaler")
    print(f"   Parameters: {total_params:,}")
    print(f"   Model file: {model_file}")
    print(f"   Scalers file: {scalers_file}")
    
    return mae, r2, exact_matches

if __name__ == "__main__":
    evaluate_robustscaler_ultraresnet() 