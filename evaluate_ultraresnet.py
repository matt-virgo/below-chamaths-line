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

def evaluate_ultraresnet_model():
    """Load and evaluate the newly trained UltraResNet model"""
    
    print("🎯 Evaluating UltraResNet Model")
    print("=" * 50)
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra features...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Load the saved scalers
    print("Loading saved scalers...")
    try:
        with open('ultraresnet_scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            scaler_X = scalers['scaler_X']
            scaler_y = scalers['scaler_y']
        print("✅ Scalers loaded successfully")
    except FileNotFoundError:
        print("❌ ultraresnet_scalers.pkl not found!")
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
    print("\nLoading trained UltraResNet model...")
    try:
        model.load_state_dict(torch.load('ultraresnet_final_model.pth', map_location='cpu'))
        model.eval()
        print("✅ ultraresnet_final_model.pth loaded successfully!")
    except FileNotFoundError:
        print("❌ ultraresnet_final_model.pth not found!")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Evaluate on test set
    print("\n=== ULTRARESNET MODEL EVALUATION ===")
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
    
    print(f"🎯 UltraResNet Performance:")
    print(f"   Test MAE: ${mae:.2f}")
    print(f"   Test RMSE: ${rmse:.2f}")
    print(f"   Test R²: {r2:.6f}")
    print(f"   Exact matches (±$0.01): {exact_matches}/{len(actuals)} ({exact_matches/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$1.00): {close_matches_1}/{len(actuals)} ({close_matches_1/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$5.00): {close_matches_5}/{len(actuals)} ({close_matches_5/len(actuals)*100:.1f}%)")
    print(f"   Close matches (±$10.00): {close_matches_10}/{len(actuals)} ({close_matches_10/len(actuals)*100:.1f}%)")
    
    # Compare to benchmarks
    print(f"\n📊 Performance Comparison:")
    print(f"   Target (original UltraResNet): $58.91 MAE")
    print(f"   Current UltraResNet: ${mae:.2f} MAE")
    print(f"   Latest aggressive model: $68.75 MAE")
    
    if mae < 58.91:
        improvement = 58.91 - mae
        print(f"   🎯 NEW RECORD: ${improvement:.2f} better than original!")
    elif abs(mae - 58.91) < 2.0:
        difference = mae - 58.91
        print(f"   ✅ VERY CLOSE: Only ${difference:.2f} away from original")
    else:
        difference = mae - 58.91
        print(f"   ⚠️  Gap: ${difference:.2f} away from original")
    
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
    
    # Save predictions for analysis
    results_df = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': actuals,
        'predicted_reimbursement': predictions,
        'error': actuals - predictions,
        'abs_error': errors
    })
    
    results_df.to_csv('ultraresnet_evaluation_results.csv', index=False)
    print(f"\n💾 Evaluation results saved to: ultraresnet_evaluation_results.csv")
    
    return mae, r2, exact_matches

if __name__ == "__main__":
    evaluate_ultraresnet_model() 