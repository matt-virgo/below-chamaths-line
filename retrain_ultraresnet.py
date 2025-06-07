#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import the model architecture and features from ultra_deep_learning
from ultra_deep_learning import UltraResNet, create_ultra_features, load_data

def train_model(model, train_loader, val_loader, epochs=2000, lr=0.001, weight_decay=1e-3, patience=100):
    """Train model with cosine annealing and early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        cosine_scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'ultraresnet_58_91_checkpoint.pth')
        else:
            patience_counter += 1
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('ultraresnet_58_91_checkpoint.pth'))
    return model

def evaluate_model(model, data_loader, scaler_y=None):
    """Evaluate model performance with detailed metrics"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Reverse scaling if needed
    if scaler_y is not None:
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    return predictions, actuals, mae, rmse, r2

def main():
    """Re-train the UltraResNet model that achieved $58.91 MAE"""
    
    print("🎯 Retraining UltraResNet Model for $58.91 MAE")
    print("=" * 60)
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra-comprehensive features (58 features)...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Use RobustScaler (exact configuration that achieved $58.91)
    print("Applying RobustScaler...")
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled), 
        torch.FloatTensor(y_train_scaled)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled), 
        torch.FloatTensor(y_test_scaled)
    )
    
    # Split training data for validation (85% train, 15% val)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    input_size = X_train_scaled.shape[1]
    
    print(f"\nCreating UltraResNet model...")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: 256")
    print(f"   Residual blocks: 8")
    print(f"   Weight decay: 1e-3")
    
    # Create UltraResNet with exact configuration that achieved $58.91 MAE
    model = UltraResNet(input_size=input_size, hidden_size=256, num_blocks=8)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    print(f"\n🚀 Starting training...")
    trained_model = train_model(
        model, train_loader, val_loader, 
        epochs=2000, lr=0.001, weight_decay=1e-3, patience=100
    )
    
    print(f"\n📊 Evaluating trained model...")
    
    # Evaluate on training set
    train_pred, train_actual, train_mae, train_rmse, train_r2 = evaluate_model(
        trained_model, train_loader, scaler_y
    )
    
    # Evaluate on test set
    test_pred, test_actual, test_mae, test_rmse, test_r2 = evaluate_model(
        trained_model, test_loader, scaler_y
    )
    
    # Calculate precision metrics
    exact_matches = np.sum(np.abs(test_actual - test_pred) < 0.01)
    close_matches_1 = np.sum(np.abs(test_actual - test_pred) < 1.0)
    close_matches_5 = np.sum(np.abs(test_actual - test_pred) < 5.0)
    close_matches_10 = np.sum(np.abs(test_actual - test_pred) < 10.0)
    
    print(f"\n🎯 FINAL RESULTS:")
    print(f"   Train MAE: ${train_mae:.2f}")
    print(f"   Test MAE:  ${test_mae:.2f}")
    print(f"   Train R²:  {train_r2:.6f}")
    print(f"   Test R²:   {test_r2:.6f}")
    print(f"   Exact matches (±$0.01): {exact_matches}/{len(test_actual)} ({exact_matches/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$1.00): {close_matches_1}/{len(test_actual)} ({close_matches_1/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$5.00): {close_matches_5}/{len(test_actual)} ({close_matches_5/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$10.00): {close_matches_10}/{len(test_actual)} ({close_matches_10/len(test_actual)*100:.1f}%)")
    
    # Check if we achieved the target performance
    if abs(test_mae - 58.91) < 1.0:
        print(f"\n✅ SUCCESS: Achieved target performance (within $1.00 of $58.91)!")
    elif test_mae < 58.91:
        improvement = 58.91 - test_mae
        print(f"\n🎯 EXCELLENT: Beat target by ${improvement:.2f}!")
    else:
        difference = test_mae - 58.91
        print(f"\n⚠️  Close: ${difference:.2f} away from target $58.91")
    
    # Save the final model with clear naming
    final_model_path = 'ultraresnet_final_model.pth'
    torch.save(trained_model.state_dict(), final_model_path)
    
    # Also save the scalers for later use
    import pickle
    with open('ultraresnet_scalers.pkl', 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    
    print(f"\n💾 Model saved:")
    print(f"   Model weights: {final_model_path}")
    print(f"   Scalers: ultraresnet_scalers.pkl")
    print(f"   Checkpoint: ultraresnet_58_91_checkpoint.pth")
    
    # Show best predictions
    errors = np.abs(test_actual - test_pred)
    best_indices = np.argsort(errors)[:10]
    
    print(f"\n🏆 Top 10 most accurate predictions:")
    for i, idx in enumerate(best_indices):
        print(f"{i+1:2d}. Days: {test_df.iloc[idx]['trip_duration_days']:2.0f}, "
              f"Miles: {test_df.iloc[idx]['miles_traveled']:4.0f}, "
              f"Receipts: ${test_df.iloc[idx]['total_receipts_amount']:7.2f}, "
              f"Actual: ${test_actual[idx]:7.2f}, "
              f"Predicted: ${test_pred[idx]:7.2f}, "
              f"Error: ${errors[idx]:.4f}")
    
    return test_mae, test_r2, exact_matches

if __name__ == "__main__":
    main() 