#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_data():
    """Load training and test data"""
    with open('train_cases.json', 'r') as f:
        train_data = json.load(f)
    
    with open('test_cases.json', 'r') as f:
        test_data = json.load(f)
    
    # Convert to DataFrames
    train_df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        for case in train_data
    ])
    
    test_df = pd.DataFrame([
        {
            'trip_duration_days': case['input']['trip_duration_days'],
            'miles_traveled': case['input']['miles_traveled'],
            'total_receipts_amount': case['input']['total_receipts_amount'],
            'reimbursement': case['expected_output']
        }
        for case in test_data
    ])
    
    return train_df, test_df

def create_ultra_features(df):
    """Create the optimal 58-feature set from the best model"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # Core derived features
    features_df['miles_per_day'] = M / D
    features_df['receipts_per_day'] = R / D
    
    # Most important validated features
    features_df['total_trip_value'] = D * M * R
    features_df['receipts_log'] = np.log1p(R)
    features_df['receipts_sqrt'] = np.sqrt(R)
    features_df['receipts_squared'] = R ** 2
    features_df['receipts_cubed'] = R ** 3
    
    # Miles transformations
    features_df['miles_log'] = np.log1p(M)
    features_df['miles_sqrt'] = np.sqrt(M)
    features_df['miles_squared'] = M ** 2
    features_df['miles_cubed'] = M ** 3
    
    # Days transformations
    features_df['days_squared'] = D ** 2
    features_df['days_cubed'] = D ** 3
    features_df['days_fourth'] = D ** 4
    
    # Lucky cents feature
    features_df['receipts_cents'] = (R * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(float)
    
    # Comprehensive interactions
    features_df['miles_receipts'] = M * R
    features_df['days_receipts'] = D * R
    features_df['days_miles'] = D * M
    features_df['miles_per_day_squared'] = features_df['miles_per_day'] ** 2
    features_df['receipts_per_day_squared'] = features_df['receipts_per_day'] ** 2
    features_df['miles_receipts_per_day'] = features_df['miles_per_day'] * features_df['receipts_per_day']
    
    # Complex ratio features
    features_df['receipts_to_miles_ratio'] = R / (M + 1)
    features_df['miles_to_days_ratio'] = M / D
    features_df['total_value_per_day'] = features_df['total_trip_value'] / D
    
    # Trigonometric features (for cyclical patterns)
    features_df['receipts_sin_1000'] = np.sin(R / 1000)
    features_df['receipts_cos_1000'] = np.cos(R / 1000)
    features_df['receipts_sin_500'] = np.sin(R / 500)
    features_df['receipts_cos_500'] = np.cos(R / 500)
    features_df['miles_sin_500'] = np.sin(M / 500)
    features_df['miles_cos_500'] = np.cos(M / 500)
    features_df['miles_sin_1000'] = np.sin(M / 1000)
    features_df['miles_cos_1000'] = np.cos(M / 1000)
    
    # Exponential features (normalized)
    features_df['receipts_exp_norm'] = np.exp(R / 2000) - 1
    features_df['miles_exp_norm'] = np.exp(M / 1000) - 1
    features_df['days_exp_norm'] = np.exp(D / 10) - 1
    
    # Polynomial combinations
    features_df['days_miles_receipts'] = D * M * R
    features_df['sqrt_days_miles_receipts'] = np.sqrt(D * M * R)
    features_df['log_days_miles_receipts'] = np.log1p(D * M * R)
    
    # High-order interactions
    features_df['d2_m_r'] = (D ** 2) * M * R
    features_df['d_m2_r'] = D * (M ** 2) * R
    features_df['d_m_r2'] = D * M * (R ** 2)
    
    # Binned features (for threshold detection)
    features_df['receipts_bin_20'] = pd.cut(R, bins=20, labels=False)
    features_df['miles_bin_20'] = pd.cut(M, bins=20, labels=False)
    features_df['days_bin_10'] = pd.cut(D, bins=10, labels=False)
    
    # Per-day thresholds
    mpd = M / D
    rpd = R / D
    features_df['mpd_low'] = (mpd < 100).astype(float)
    features_df['mpd_med'] = ((mpd >= 100) & (mpd <= 200)).astype(float)
    features_df['mpd_high'] = (mpd > 200).astype(float)
    features_df['rpd_low'] = (rpd < 75).astype(float)
    features_df['rpd_med'] = ((rpd >= 75) & (rpd <= 150)).astype(float)
    features_df['rpd_high'] = (rpd > 150).astype(float)
    
    # Special case indicators
    features_df['is_short_trip'] = (D <= 2).astype(float)
    features_df['is_medium_trip'] = ((D >= 3) & (D <= 7)).astype(float)
    features_df['is_long_trip'] = (D >= 8).astype(float)
    features_df['is_5_day_trip'] = (D == 5).astype(float)
    
    # Remove target if present
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    return features_df[feature_cols]

class AggressiveResidualBlock(nn.Module):
    """Enhanced residual block with very aggressive regularization"""
    
    def __init__(self, size, dropout_rate=0.15):
        super(AggressiveResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.Dropout(dropout_rate * 0.7)
        )
        self.relu = nn.ReLU()
        self.final_dropout = nn.Dropout(dropout_rate * 0.4)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return self.final_dropout(out)

class AggressiveUltraResNet(nn.Module):
    """Ultra-deep ResNet with aggressive regularization for extended training"""
    
    def __init__(self, input_size, hidden_size=256, num_blocks=12, dropout_rate=0.15):
        super(AggressiveUltraResNet, self).__init__()
        
        # Enhanced input layer with stronger regularization
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 2.5)
        )
        
        # More residual blocks for increased capacity
        self.blocks = nn.ModuleList([
            AggressiveResidualBlock(hidden_size, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Enhanced output layers with gradual reduction and heavy regularization
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate * 3),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

def train_aggressive_model(model, train_loader, val_loader, epochs=10000, lr=0.0003, weight_decay=5e-3, patience=300):
    """Train with ultra-aggressive regularization and extended training"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on device: {device}")
    
    # Enhanced loss function with Label Smoothing effect
    criterion = nn.SmoothL1Loss(beta=0.1)
    
    # Ultra-aggressive optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.input_layer.parameters(), 'lr': lr * 0.3},   # Much slower for input
        {'params': model.blocks.parameters(), 'lr': lr},               # Normal for blocks
        {'params': model.output_layer.parameters(), 'lr': lr * 2}     # Faster for output
    ], weight_decay=weight_decay, eps=1e-8, amsgrad=True)
    
    # Multiple advanced schedulers
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=0.6, min_lr=1e-8
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-8
    )
    
    # Exponential scheduler for very gradual decay
    exp_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
    
    # Enhanced training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"Starting ultra-aggressive training for {epochs} epochs...")
    print(f"Weight decay: {weight_decay}")
    print(f"Initial learning rate: {lr}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Ultra-aggressive L1 regularization
            l1_lambda = 5e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            # Ultra-aggressive L2 regularization (additional to weight_decay)
            l2_lambda = 1e-5
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                val_batches += 1
        
        train_loss /= num_batches
        val_loss /= val_batches
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update all schedulers
        plateau_scheduler.step(val_loss)
        cosine_scheduler.step()
        exp_scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'aggressive_best_model.pth')
        else:
            patience_counter += 1
        
        # Enhanced logging
        if epoch % 100 == 0 or epoch < 20:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.9f}, Best Val: {best_val_loss:.6f}, Patience: {patience_counter}")
        
        # Enhanced early stopping with more patience
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience: {patience})")
            break
    
    # Load best model
    model.load_state_dict(torch.load('aggressive_best_model.pth'))
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return model

def evaluate_aggressive_model(model, data_loader, scaler_y=None):
    """Evaluate the aggressive model with detailed metrics"""
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
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating optimal 58-feature set...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features (optimal set)")
    
    # Use RobustScaler (proven best performer)
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
    
    # Split training data for validation (use 92% for training to maximize data)
    train_size = int(0.92 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Optimized batch size
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    input_size = X_train_scaled.shape[1]
    
    # Create the aggressive model
    print(f"\nCreating Aggressive UltraResNet with {input_size} input features...")
    model = AggressiveUltraResNet(
        input_size=input_size, 
        hidden_size=256,        # Proven optimal size
        num_blocks=12,          # More blocks for capacity
        dropout_rate=0.15       # Aggressive dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train with ultra-aggressive settings
    trained_model = train_aggressive_model(
        model, train_loader, val_loader, 
        epochs=10000,           # Much longer training
        lr=0.0003,              # Conservative learning rate
        weight_decay=5e-3,      # Ultra-aggressive weight decay (5x stronger)
        patience=300            # More patience for convergence
    )
    
    # Evaluate on test set
    print("\n=== AGGRESSIVE MODEL EVALUATION ===")
    test_pred, test_actual, test_mae, test_rmse, test_r2 = evaluate_aggressive_model(
        trained_model, test_loader, scaler_y
    )
    
    # Calculate precision metrics
    exact_matches = np.sum(np.abs(test_actual - test_pred) < 0.01)
    close_matches_1 = np.sum(np.abs(test_actual - test_pred) < 1.0)
    close_matches_5 = np.sum(np.abs(test_actual - test_pred) < 5.0)
    close_matches_10 = np.sum(np.abs(test_actual - test_pred) < 10.0)
    
    print(f"🚀 Ultra-Aggressive Training Results:")
    print(f"   Test MAE: ${test_mae:.2f}")
    print(f"   Test RMSE: ${test_rmse:.2f}")
    print(f"   Test R²: {test_r2:.6f}")
    print(f"   Exact matches (±$0.01): {exact_matches}/{len(test_actual)} ({exact_matches/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$1.00): {close_matches_1}/{len(test_actual)} ({close_matches_1/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$5.00): {close_matches_5}/{len(test_actual)} ({close_matches_5/len(test_actual)*100:.1f}%)")
    print(f"   Close matches (±$10.00): {close_matches_10}/{len(test_actual)} ({close_matches_10/len(test_actual)*100:.1f}%)")
    
    # Compare to previous best
    print(f"\n📊 Performance Comparison:")
    print(f"   Previous best UltraResNet: $58.91 MAE")
    print(f"   Aggressive UltraResNet: ${test_mae:.2f} MAE")
    if test_mae < 58.91:
        improvement = 58.91 - test_mae
        print(f"   🎯 NEW RECORD: ${improvement:.2f} improvement!")
    else:
        difference = test_mae - 58.91
        print(f"   Still ${difference:.2f} behind previous best")
    
    # Save results
    aggressive_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'aggressive_prediction': test_pred,
        'error': test_df['reimbursement'] - test_pred,
        'abs_error': np.abs(test_df['reimbursement'] - test_pred)
    })
    
    aggressive_results.to_csv('aggressive_ultra_results.csv', index=False)
    print(f"\nResults saved to aggressive_ultra_results.csv")
    
    # Analyze the best predictions
    sorted_by_error = aggressive_results.sort_values('abs_error')
    print(f"\nTop 15 most accurate predictions:")
    for i in range(15):
        row = sorted_by_error.iloc[i]
        print(f"{i+1:2d}. Days: {row['trip_duration_days']:2.0f}, "
              f"Miles: {row['miles_traveled']:4.0f}, "
              f"Receipts: ${row['total_receipts_amount']:7.2f}, "
              f"Actual: ${row['actual_reimbursement']:7.2f}, "
              f"Predicted: ${row['aggressive_prediction']:7.2f}, "
              f"Error: ${row['abs_error']:.4f}")

if __name__ == "__main__":
    main() 