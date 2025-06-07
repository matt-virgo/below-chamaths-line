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

def create_mega_features(df):
    """Create an even more extensive feature set"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # Core derived features
    features_df['miles_per_day'] = M / D
    features_df['receipts_per_day'] = R / D
    
    # Most important validated features (from previous analysis)
    features_df['total_trip_value'] = D * M * R
    features_df['receipts_log'] = np.log1p(R)
    features_df['receipts_sqrt'] = np.sqrt(R)
    features_df['receipts_squared'] = R ** 2
    features_df['receipts_cubed'] = R ** 3
    features_df['receipts_fourth'] = R ** 4
    
    # Miles transformations (more extensive)
    features_df['miles_log'] = np.log1p(M)
    features_df['miles_sqrt'] = np.sqrt(M)
    features_df['miles_squared'] = M ** 2
    features_df['miles_cubed'] = M ** 3
    features_df['miles_fourth'] = M ** 4
    features_df['miles_fifth'] = M ** 5
    
    # Days transformations (more extensive)
    features_df['days_squared'] = D ** 2
    features_df['days_cubed'] = D ** 3
    features_df['days_fourth'] = D ** 4
    features_df['days_fifth'] = D ** 5
    features_df['days_log'] = np.log1p(D)
    features_df['days_sqrt'] = np.sqrt(D)
    
    # Lucky cents and numerical patterns
    features_df['receipts_cents'] = (R * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(float)
    features_df['receipts_ends_in_zero'] = (features_df['receipts_cents'] == 0).astype(float)
    features_df['receipts_round_number'] = (features_df['receipts_cents'] % 25 == 0).astype(float)
    
    # Miles patterns
    features_df['miles_round_100'] = (M % 100 == 0).astype(float)
    features_df['miles_round_50'] = (M % 50 == 0).astype(float)
    features_df['miles_even'] = (M % 2 == 0).astype(float)
    
    # Comprehensive interactions (expanded)
    features_df['miles_receipts'] = M * R
    features_df['days_receipts'] = D * R
    features_df['days_miles'] = D * M
    features_df['miles_per_day_squared'] = features_df['miles_per_day'] ** 2
    features_df['receipts_per_day_squared'] = features_df['receipts_per_day'] ** 2
    features_df['miles_receipts_per_day'] = features_df['miles_per_day'] * features_df['receipts_per_day']
    
    # Higher order interactions
    features_df['d2_m_r'] = (D ** 2) * M * R
    features_df['d_m2_r'] = D * (M ** 2) * R
    features_df['d_m_r2'] = D * M * (R ** 2)
    features_df['d2_m2_r'] = (D ** 2) * (M ** 2) * R
    features_df['d2_m_r2'] = (D ** 2) * M * (R ** 2)
    features_df['d_m2_r2'] = D * (M ** 2) * (R ** 2)
    features_df['d3_m_r'] = (D ** 3) * M * R
    features_df['d_m3_r'] = D * (M ** 3) * R
    features_df['d_m_r3'] = D * M * (R ** 3)
    
    # Complex ratio features
    features_df['receipts_to_miles_ratio'] = R / (M + 1)
    features_df['miles_to_days_ratio'] = M / D
    features_df['total_value_per_day'] = features_df['total_trip_value'] / D
    features_df['receipts_to_total_ratio'] = R / (D + M + R)
    features_df['miles_to_total_ratio'] = M / (D + M + R)
    features_df['days_to_total_ratio'] = D / (D + M + R)
    
    # Trigonometric features (more frequencies)
    for divisor in [100, 250, 500, 750, 1000, 1500, 2000]:
        features_df[f'receipts_sin_{divisor}'] = np.sin(R / divisor)
        features_df[f'receipts_cos_{divisor}'] = np.cos(R / divisor)
        
    for divisor in [50, 100, 250, 500, 750, 1000]:
        features_df[f'miles_sin_{divisor}'] = np.sin(M / divisor)
        features_df[f'miles_cos_{divisor}'] = np.cos(M / divisor)
        
    for divisor in [2, 5, 10, 15, 20]:
        features_df[f'days_sin_{divisor}'] = np.sin(D / divisor)
        features_df[f'days_cos_{divisor}'] = np.cos(D / divisor)
    
    # Exponential features (more scales)
    features_df['receipts_exp_norm_500'] = np.exp(R / 500) - 1
    features_df['receipts_exp_norm_1000'] = np.exp(R / 1000) - 1
    features_df['receipts_exp_norm_2000'] = np.exp(R / 2000) - 1
    features_df['receipts_exp_norm_3000'] = np.exp(R / 3000) - 1
    features_df['miles_exp_norm_250'] = np.exp(M / 250) - 1
    features_df['miles_exp_norm_500'] = np.exp(M / 500) - 1
    features_df['miles_exp_norm_1000'] = np.exp(M / 1000) - 1
    features_df['days_exp_norm_5'] = np.exp(D / 5) - 1
    features_df['days_exp_norm_10'] = np.exp(D / 10) - 1
    features_df['days_exp_norm_20'] = np.exp(D / 20) - 1
    
    # Polynomial combinations (extended)
    features_df['days_miles_receipts'] = D * M * R
    features_df['sqrt_days_miles_receipts'] = np.sqrt(D * M * R)
    features_df['log_days_miles_receipts'] = np.log1p(D * M * R)
    features_df['cbrt_days_miles_receipts'] = np.power(D * M * R, 1/3)
    features_df['fourth_root_days_miles_receipts'] = np.power(D * M * R, 1/4)
    
    # Binned features (more granular)
    features_df['receipts_bin_50'] = pd.cut(R, bins=50, labels=False)
    features_df['miles_bin_50'] = pd.cut(M, bins=50, labels=False)
    features_df['days_bin_20'] = pd.cut(D, bins=20, labels=False)
    
    # Per-day thresholds (more granular)
    mpd = M / D
    rpd = R / D
    features_df['mpd_very_low'] = (mpd < 50).astype(float)
    features_df['mpd_low'] = ((mpd >= 50) & (mpd < 100)).astype(float)
    features_df['mpd_med_low'] = ((mpd >= 100) & (mpd < 150)).astype(float)
    features_df['mpd_med'] = ((mpd >= 150) & (mpd < 200)).astype(float)
    features_df['mpd_med_high'] = ((mpd >= 200) & (mpd < 250)).astype(float)
    features_df['mpd_high'] = ((mpd >= 250) & (mpd < 300)).astype(float)
    features_df['mpd_very_high'] = (mpd >= 300).astype(float)
    
    features_df['rpd_very_low'] = (rpd < 50).astype(float)
    features_df['rpd_low'] = ((rpd >= 50) & (rpd < 75)).astype(float)
    features_df['rpd_med_low'] = ((rpd >= 75) & (rpd < 100)).astype(float)
    features_df['rpd_med'] = ((rpd >= 100) & (rpd < 125)).astype(float)
    features_df['rpd_med_high'] = ((rpd >= 125) & (rpd < 150)).astype(float)
    features_df['rpd_high'] = ((rpd >= 150) & (rpd < 200)).astype(float)
    features_df['rpd_very_high'] = (rpd >= 200).astype(float)
    
    # Special case indicators (expanded)
    features_df['is_1_day_trip'] = (D == 1).astype(float)
    features_df['is_2_day_trip'] = (D == 2).astype(float)
    features_df['is_3_day_trip'] = (D == 3).astype(float)
    features_df['is_4_day_trip'] = (D == 4).astype(float)
    features_df['is_5_day_trip'] = (D == 5).astype(float)
    features_df['is_week_trip'] = (D == 7).astype(float)
    features_df['is_two_week_trip'] = (D == 14).astype(float)
    features_df['is_weekend_trip'] = (D <= 2).astype(float)
    features_df['is_short_trip'] = (D <= 3).astype(float)
    features_df['is_medium_trip'] = ((D >= 4) & (D <= 7)).astype(float)
    features_df['is_long_trip'] = (D >= 8).astype(float)
    features_df['is_very_long_trip'] = (D >= 12).astype(float)
    
    # Business logic features (based on common reimbursement rules)
    features_df['per_diem_50'] = D * 50
    features_df['per_diem_75'] = D * 75
    features_df['per_diem_100'] = D * 100
    features_df['per_diem_125'] = D * 125
    features_df['per_diem_150'] = D * 150
    features_df['per_diem_200'] = D * 200
    
    features_df['mileage_30'] = M * 0.30
    features_df['mileage_40'] = M * 0.40
    features_df['mileage_50'] = M * 0.50
    features_df['mileage_55'] = M * 0.55
    features_df['mileage_60'] = M * 0.60
    features_df['mileage_65'] = M * 0.65
    
    features_df['receipt_percent_30'] = R * 0.30
    features_df['receipt_percent_40'] = R * 0.40
    features_df['receipt_percent_50'] = R * 0.50
    features_df['receipt_percent_60'] = R * 0.60
    features_df['receipt_percent_70'] = R * 0.70
    features_df['receipt_percent_80'] = R * 0.80
    features_df['receipt_percent_90'] = R * 0.90
    features_df['receipt_percent_100'] = R * 1.00
    
    # Combined business logic
    features_df['standard_calc_1'] = D * 100 + M * 0.50 + R * 0.50
    features_df['standard_calc_2'] = D * 125 + M * 0.55 + R * 0.60
    features_df['standard_calc_3'] = D * 75 + M * 0.40 + R * 0.70
    
    # Remove target if present
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    return features_df[feature_cols]

class MegaResidualBlock(nn.Module):
    """Enhanced residual block with even better regularization and capacity"""
    
    def __init__(self, size, dropout_rate=0.1):
        super(MegaResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.Dropout(dropout_rate * 0.5)
        )
        self.relu = nn.ReLU()
        self.final_dropout = nn.Dropout(dropout_rate * 0.3)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return self.final_dropout(out)

class MegaUltraResNet(nn.Module):
    """Massive ResNet with many more blocks and larger hidden dimensions"""
    
    def __init__(self, input_size, hidden_size=512, num_blocks=16, dropout_rate=0.1):
        super(MegaUltraResNet, self).__init__()
        
        # Much larger input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 2)
        )
        
        # Many more residual blocks
        self.blocks = nn.ModuleList([
            MegaResidualBlock(hidden_size, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Larger output layers with gradual reduction
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate * 3),
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

def train_mega_model(model, train_loader, val_loader, epochs=5000, lr=0.0005, weight_decay=1e-3, patience=200):
    """Train the mega model for much longer with enhanced techniques"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Training on device: {device}")
    
    # Enhanced loss function
    criterion = nn.SmoothL1Loss()
    
    # Optimizer with different learning rates for different parts
    optimizer = optim.AdamW([
        {'params': model.input_layer.parameters(), 'lr': lr * 0.5},  # Slower for input
        {'params': model.blocks.parameters(), 'lr': lr},              # Normal for blocks
        {'params': model.output_layer.parameters(), 'lr': lr * 1.5}  # Faster for output
    ], weight_decay=weight_decay, eps=1e-8)
    
    # Multiple schedulers
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=50, factor=0.7, min_lr=1e-7
    )
    
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=100, T_mult=2, eta_min=1e-7
    )
    
    # Enhanced training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
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
            
            # Enhanced L1 regularization
            l1_lambda = 2e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
        
        # Update schedulers
        plateau_scheduler.step(val_loss)
        cosine_scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'mega_best_model.pth')
        else:
            patience_counter += 1
        
        # Logging
        if epoch % 50 == 0 or epoch < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:4d}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                  f"LR: {current_lr:.8f}, Best Val: {best_val_loss:.6f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('mega_best_model.pth'))
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return model

def evaluate_mega_model(model, data_loader, scaler_y=None):
    """Evaluate the mega model with detailed metrics"""
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
    
    print("Creating mega-comprehensive features...")
    X_train = create_mega_features(train_df)
    X_test = create_mega_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features (massive increase!)")
    
    # Use RobustScaler (best performer from previous analysis)
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
    
    # Split training data for validation (use less for validation to get more training data)
    train_size = int(0.9 * len(train_dataset))  # Use 90% for training
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Smaller batch size for the massive model
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = X_train_scaled.shape[1]
    
    # Create the mega model
    print(f"\nCreating Mega UltraResNet with {input_size} input features...")
    model = MegaUltraResNet(
        input_size=input_size, 
        hidden_size=512,        # Much larger
        num_blocks=16,          # Many more blocks
        dropout_rate=0.08       # Fine-tuned dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train the mega model
    trained_model = train_mega_model(
        model, train_loader, val_loader, 
        epochs=5000,            # Much longer training
        lr=0.0005,              # Slower learning rate
        weight_decay=1e-3,      # Strong regularization
        patience=200            # More patience
    )
    
    # Evaluate on test set
    print("\n=== MEGA MODEL EVALUATION ===")
    test_pred, test_actual, test_mae, test_rmse, test_r2 = evaluate_mega_model(
        trained_model, test_loader, scaler_y
    )
    
    # Calculate precision metrics
    exact_matches = np.sum(np.abs(test_actual - test_pred) < 0.01)
    close_matches_1 = np.sum(np.abs(test_actual - test_pred) < 1.0)
    close_matches_5 = np.sum(np.abs(test_actual - test_pred) < 5.0)
    close_matches_10 = np.sum(np.abs(test_actual - test_pred) < 10.0)
    
    print(f"Test MAE: ${test_mae:.2f}")
    print(f"Test RMSE: ${test_rmse:.2f}")
    print(f"Test R²: {test_r2:.6f}")
    print(f"Exact matches (±$0.01): {exact_matches}/{len(test_actual)} ({exact_matches/len(test_actual)*100:.1f}%)")
    print(f"Close matches (±$1.00): {close_matches_1}/{len(test_actual)} ({close_matches_1/len(test_actual)*100:.1f}%)")
    print(f"Close matches (±$5.00): {close_matches_5}/{len(test_actual)} ({close_matches_5/len(test_actual)*100:.1f}%)")
    print(f"Close matches (±$10.00): {close_matches_10}/{len(test_actual)} ({close_matches_10/len(test_actual)*100:.1f}%)")
    
    # Save results
    mega_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'mega_prediction': test_pred,
        'error': test_df['reimbursement'] - test_pred,
        'abs_error': np.abs(test_df['reimbursement'] - test_pred)
    })
    
    mega_results.to_csv('mega_ultra_results.csv', index=False)
    print(f"\nMega results saved to mega_ultra_results.csv")
    
    # Analyze the best predictions
    sorted_by_error = mega_results.sort_values('abs_error')
    print(f"\nTop 15 most accurate predictions:")
    for i in range(15):
        row = sorted_by_error.iloc[i]
        print(f"{i+1:2d}. Days: {row['trip_duration_days']:2.0f}, "
              f"Miles: {row['miles_traveled']:4.0f}, "
              f"Receipts: ${row['total_receipts_amount']:7.2f}, "
              f"Actual: ${row['actual_reimbursement']:7.2f}, "
              f"Predicted: ${row['mega_prediction']:7.2f}, "
              f"Error: ${row['abs_error']:.4f}")

if __name__ == "__main__":
    main() 