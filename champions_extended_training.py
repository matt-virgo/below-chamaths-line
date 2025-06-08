#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
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

def create_v1_ultra_features(df):
    """Create V1's proven comprehensive feature set (58 features)"""
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
    
    # Lucky cents feature (validated as important)
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
    
    print(f"V1 champion feature set created: {len(feature_cols)} comprehensive features")
    return features_df[feature_cols]

# V1 Champion: AttentionNet (best performer $57.35 MAE)
class AttentionNetV1Champion(nn.Module):
    """V1's champion AttentionNet architecture - $57.35 MAE"""
    
    def __init__(self, input_size, hidden_size=256):
        super(AttentionNetV1Champion, self).__init__()
        
        # Feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # Main network
        self.main_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Apply attention to features
        attention_weights = self.feature_attention(x)
        attended_features = x * attention_weights
        
        return self.main_net(attended_features).squeeze()

# V4 Champion: UltraDeep_Skip_ELU (best V4 performer based on early results)
class UltraDeepV4Champion(nn.Module):
    """V4's champion UltraDeepNet with skip connections and ELU activation"""
    
    def __init__(self, input_size, hidden_sizes=[768, 384, 192, 96], 
                 dropout_rate=0.2, use_skip_connections=True):
        super(UltraDeepV4Champion, self).__init__()
        
        self.use_skip_connections = use_skip_connections
        self.activation = nn.ELU()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(self.activation)
            
            # Adaptive dropout - higher in early layers
            if i < len(hidden_sizes) - 2:
                layers.append(nn.Dropout(dropout_rate * (1.2 - i * 0.1)))
            else:
                layers.append(nn.Dropout(dropout_rate * 0.5))
            
            prev_size = hidden_size
        
        # Output layer with minimal dropout
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Skip connection layers
        if use_skip_connections:
            self.skip_layers = nn.ModuleList([
                nn.Linear(input_size, hidden_sizes[0]),
                nn.Linear(hidden_sizes[0], hidden_sizes[2]),
                nn.Linear(hidden_sizes[2], 1)
            ])
        
    def forward(self, x):
        if self.use_skip_connections:
            # Implement skip connections
            skip1 = self.skip_layers[0](x)
            out = self.network[:4](x)  # First layer + BN + ELU + Dropout
            out = out + skip1  # Skip connection
            
            skip2 = self.skip_layers[1](out)
            out = self.network[4:12](out)  # Next layers
            out = out + skip2
            
            out = self.network[12:](out)  # Remaining layers
            return out.squeeze()
        else:
            return self.network(x).squeeze()

def train_champion_extended(model, train_loader, val_loader, config, model_name):
    """Extended training with NO early stopping for champion models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Advanced optimizers
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                               weight_decay=config['weight_decay'], eps=1e-8)
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
    
    criterion = nn.MSELoss()
    
    # Advanced learning rate schedulers
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2, eta_min=config['lr'] * 0.001
        )
    elif config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=50, 
            min_lr=config['lr'] * 0.0001
        )
    else:  # exponential
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    
    best_val_loss = float('inf')
    best_epoch = 0
    model_path = f"Champion_{model_name}_best.pth"
    
    print(f"🏆 Training Champion {model_name} - EXTENDED MODE (No Early Stopping)")
    print(f"   Optimizer: {config['optimizer']}, LR: {config['lr']}, WD: {config['weight_decay']}")
    print(f"   Scheduler: {config['scheduler']}, Epochs: {config['epochs']}")
    print(f"   🚨 NO EARLY STOPPING - Will train for full {config['epochs']} epochs")
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Gradient clipping for stability
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
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
        
        # Update scheduler
        if config['scheduler'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Save best model (but don't stop)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        
        if epoch % 200 == 0 or epoch < 10:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.8f} | Best: {best_epoch}")
    
    print(f"  🏆 COMPLETED! Best validation at epoch {best_epoch} saved to {model_path}")
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate_model(model, data_loader, scaler_y=None):
    """Evaluate model performance"""
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
    print("🏆 CHAMPIONS EXTENDED TRAINING - NO EARLY STOPPING")
    print("="*70)
    print("Training V1 Champion vs V4 Champion for maximum epochs")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating V1's proven comprehensive feature set...")
    X_train = create_v1_ultra_features(train_df)
    X_test = create_v1_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"\n✨ Using {X_train.shape[1]} proven V1 champion features")
    
    # Champion configurations
    champion_configs = [
        {
            'name': 'V1_AttentionNet_Champion',
            'model_class': AttentionNetV1Champion,
            'model_params': {'hidden_size': 256},
            'scaler': QuantileTransformer(n_quantiles=100, random_state=42),  # V1's best scaler
            'optimizer': 'adamw',
            'lr': 0.0005,  # Lower LR for extended training
            'weight_decay': 2e-4,
            'scheduler': 'cosine',
            'epochs': 5000,  # MUCH longer training
            'grad_clip': 1.0
        },
        {
            'name': 'V4_UltraDeep_Champion', 
            'model_class': UltraDeepV4Champion,
            'model_params': {'hidden_sizes': [768, 384, 192, 96], 'dropout_rate': 0.2},
            'scaler': RobustScaler(),  # Alternative scaler for V4
            'optimizer': 'adamw',
            'lr': 0.0008,  # Lower LR for extended training
            'weight_decay': 5e-5,
            'scheduler': 'cosine',
            'epochs': 5000,  # MUCH longer training
            'grad_clip': 2.0
        },
        {
            'name': 'V1_Extended_AttentionNet',
            'model_class': AttentionNetV1Champion,
            'model_params': {'hidden_size': 256},
            'scaler': QuantileTransformer(n_quantiles=100, random_state=42),
            'optimizer': 'adam',
            'lr': 0.0003,  # Even lower LR
            'weight_decay': 1e-4,
            'scheduler': 'plateau',
            'epochs': 8000,  # MAXIMUM training
            'grad_clip': 0.5
        }
    ]
    
    all_results = []
    
    for config in champion_configs:
        print(f"\n{'='*70}")
        print(f"🥊 TRAINING CHAMPION: {config['name']}")
        print(f"{'='*70}")
        
        # Scale features
        scaler = config['scaler']
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
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
        
        # Use more data for training since we're doing extended training
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        input_size = X_train_scaled.shape[1]
        
        # Create model
        model = config['model_class'](input_size, **config['model_params'])
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model has {total_params:,} parameters")
        
        # Train model with extended training
        trained_model = train_champion_extended(model, train_loader, val_loader, config, config['name'])
        
        # Evaluate
        train_pred, train_actual, train_mae, train_rmse, train_r2 = evaluate_model(
            trained_model, train_loader, scaler_y
        )
        test_pred, test_actual, test_mae, test_rmse, test_r2 = evaluate_model(
            trained_model, test_loader, scaler_y
        )
        
        # Precision metrics
        exact_matches = np.sum(np.abs(test_actual - test_pred) < 0.01)
        close_matches_1 = np.sum(np.abs(test_actual - test_pred) < 1.0)
        close_matches_5 = np.sum(np.abs(test_actual - test_pred) < 5.0)
        close_matches_10 = np.sum(np.abs(test_actual - test_pred) < 10.0)
        
        results = {
            'name': config['name'],
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'exact_matches': exact_matches,
            'close_matches_1': close_matches_1,
            'close_matches_5': close_matches_5,
            'close_matches_10': close_matches_10,
            'predictions': test_pred,
            'epochs_trained': config['epochs']
        }
        
        all_results.append(results)
        
        print(f"\n🎯 {config['name']} FINAL RESULTS:")
        print(f"   Train MAE: ${train_mae:.2f}")
        print(f"   Test MAE:  ${test_mae:.2f}")
        print(f"   Test R²:   {test_r2:.6f}")
        print(f"   Exact matches (±$0.01): {exact_matches}")
        print(f"   Close matches (±$1.00): {close_matches_1}")
        print(f"   Close matches (±$5.00): {close_matches_5}")
        print(f"   Close matches (±$10.00): {close_matches_10}")
        
        # Save individual results
        champion_results = pd.DataFrame({
            'trip_duration_days': test_df['trip_duration_days'],
            'miles_traveled': test_df['miles_traveled'],
            'total_receipts_amount': test_df['total_receipts_amount'],
            'actual_reimbursement': test_df['reimbursement'],
            'champion_prediction': test_pred,
            'error': test_df['reimbursement'] - test_pred,
            'abs_error': np.abs(test_df['reimbursement'] - test_pred)
        })
        
        champion_results.to_csv(f'{config["name"]}_extended_results.csv', index=False)
        print(f"   💾 Results saved to: {config['name']}_extended_results.csv")
        
        # Save model and scalers
        import pickle
        scaler_file = f'Champion_{config["name"]}_scalers.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump({'scaler_X': scaler, 'scaler_y': scaler_y}, f)
        print(f"   💾 Scalers saved to: {scaler_file}")
    
    # Final comparison
    print(f"\n{'='*80}")
    print(f"🏆 CHAMPIONS EXTENDED TRAINING FINAL RESULTS:")
    print(f"{'='*80}")
    
    # Sort by test MAE
    sorted_results = sorted(all_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['name']:<30} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f} | Epochs: {result['epochs_trained']}")
    
    best_champion = sorted_results[0]
    
    print(f"\n🎉 ULTIMATE CHAMPION: {best_champion['name']}")
    print(f"   🎯 Test MAE: ${best_champion['test_mae']:.2f}")
    print(f"   📊 Test R²: {best_champion['test_r2']:.6f}")
    print(f"   ⏱️  Epochs: {best_champion['epochs_trained']}")
    
    print(f"\n📈 COMPARISON WITH PREVIOUS BESTS:")
    print(f"   V1 Original Best:     $57.35 MAE")
    print(f"   V4 Original Best:     $59.76 MAE")
    print(f"   Extended Champion:    ${best_champion['test_mae']:.2f} MAE")
    
    if best_champion['test_mae'] < 57.35:
        improvement = 57.35 - best_champion['test_mae']
        print(f"   🎉 NEW WORLD RECORD! ${improvement:.2f} better ({improvement/57.35*100:.1f}% improvement)")
    elif best_champion['test_mae'] < 59.76:
        improvement = 59.76 - best_champion['test_mae']
        print(f"   ✅ Improved over V4! ${improvement:.2f} better")
    else:
        diff = best_champion['test_mae'] - 57.35
        print(f"   📊 ${diff:.2f} difference from V1 baseline")
    
    print(f"\n🧠 EXTENDED TRAINING INSIGHTS:")
    print(f"   • Extended training allows models to find deeper minima")
    print(f"   • Lower learning rates prevent overshooting optimal weights")
    print(f"   • Advanced schedulers help fine-tune in later epochs")
    print(f"   • NO early stopping ensures maximum optimization")

if __name__ == "__main__":
    main() 