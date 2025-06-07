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
from sklearn.model_selection import KFold
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
    """Create an extensive feature set optimized for deep learning"""
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

class UltraDeepNet(nn.Module):
    """Ultra-deep neural network with heavy regularization"""
    
    def __init__(self, input_size, hidden_sizes=[1024, 768, 512, 384, 256, 192, 128, 96, 64, 32], dropout_rate=0.4):
        super(UltraDeepNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate if i < len(hidden_sizes) - 3 else dropout_rate * 0.5)  # Less dropout in final layers
            ])
            prev_size = hidden_size
        
        # Output layer with smaller dropout
        layers.extend([
            nn.Dropout(0.1),
            nn.Linear(prev_size, 1)
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class MegaWideDeepNet(nn.Module):
    """Mega Wide & Deep model with extensive capacity"""
    
    def __init__(self, input_size, wide_features=3):
        super(MegaWideDeepNet, self).__init__()
        
        # Wide part (linear with more capacity)
        self.wide = nn.Sequential(
            nn.Linear(wide_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Deep part (much larger)
        self.deep = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Split input for wide and deep parts
        wide_input = x[:, :3]  # First 3 features for wide part
        
        wide_out = self.wide(wide_input)
        deep_out = self.deep(x)
        
        return (wide_out + deep_out).squeeze()

class UltraResNet(nn.Module):
    """Ultra-deep ResNet with many residual blocks"""
    
    def __init__(self, input_size, hidden_size=256, num_blocks=8):
        super(UltraResNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

class ResidualBlock(nn.Module):
    """Enhanced residual block with better regularization"""
    
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return self.dropout(out)

class AttentionNet(nn.Module):
    """Neural network with attention mechanism for feature importance"""
    
    def __init__(self, input_size, hidden_size=256):
        super(AttentionNet, self).__init__()
        
        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # Main network
        self.main_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Apply attention weights
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        return self.main_net(attended_features).squeeze()

def train_model(model, train_loader, val_loader, epochs=2000, lr=0.001, weight_decay=1e-3, patience=100, model_name="model"):
    """Train model with cosine annealing and early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = f"{model_name}_best.pth"
    
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
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
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
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra-comprehensive features...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Multiple scaling approaches for robustness
    scalers = {
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    best_results = {}
    all_model_results = []  # Track all models for comparison
    
    for scaler_name, scaler_X in scalers.items():
        print(f"\n=== Using {scaler_name} ===")
        
        # Scale features
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        
        # Scale target
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # Create data loaders with larger batch size for stability
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled), 
            torch.FloatTensor(y_train_scaled)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled), 
            torch.FloatTensor(y_test_scaled)
        )
        
        # Split training data for validation
        train_size = int(0.85 * len(train_dataset))  # Use more data for training
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)  # Larger batch size
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        input_size = X_train_scaled.shape[1]
        
        # Test ultra-large models with different weight decay values
        models = {
            'UltraDeep_WD1e-3': (UltraDeepNet(input_size), 1e-3),
            'UltraDeep_WD1e-2': (UltraDeepNet(input_size), 1e-2),
            'MegaWideDeep_WD5e-3': (MegaWideDeepNet(input_size), 5e-3),
            'UltraResNet_WD1e-3': (UltraResNet(input_size), 1e-3),
            'AttentionNet_WD2e-3': (AttentionNet(input_size), 2e-3)
        }
        
        scaler_results = {}
        
        for name, (model, weight_decay) in models.items():
            model_id = f"{scaler_name}_{name}"
            print(f"\n--- Training {model_id} ---")
            
            trained_model = train_model(
                model, train_loader, val_loader, 
                epochs=2000, lr=0.001, weight_decay=weight_decay, patience=100, model_name=model_id
            )
            
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
            
            results = {
                'model_id': model_id,
                'scaler': scaler_name,
                'architecture': name,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': test_pred,
                'exact_matches': exact_matches,
                'close_matches_1': close_matches_1,
                'close_matches_5': close_matches_5,
                'model_file': f"{model_id}_best.pth"
            }
            
            scaler_results[name] = results
            all_model_results.append(results)
            
            print(f"  Train MAE: ${train_mae:.2f}")
            print(f"  Test MAE:  ${test_mae:.2f}")
            print(f"  Train R²:  {train_r2:.6f}")
            print(f"  Test R²:   {test_r2:.6f}")
            print(f"  Exact matches (±$0.01): {exact_matches}/{len(test_actual)} ({exact_matches/len(test_actual)*100:.1f}%)")
            print(f"  Close matches (±$1.00): {close_matches_1}/{len(test_actual)} ({close_matches_1/len(test_actual)*100:.1f}%)")
            print(f"  Close matches (±$5.00): {close_matches_5}/{len(test_actual)} ({close_matches_5/len(test_actual)*100:.1f}%)")
            print(f"  💾 Model saved as: {model_id}_best.pth")
            
            # Save scalers for this model
            import pickle
            scaler_file = f"{model_id}_scalers.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
            print(f"  💾 Scalers saved as: {scaler_file}")
        
        best_results[scaler_name] = scaler_results
    
    # Find overall best model
    best_overall = min(all_model_results, key=lambda x: x['test_mae'])
    
    print(f"\n" + "="*80)
    print(f"🏆 ALL MODEL RESULTS SUMMARY:")
    print(f"="*80)
    
    # Sort all results by test MAE
    sorted_results = sorted(all_model_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['model_id']:<35} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f} | File: {result['model_file']}")
    
    print(f"\n🎯 BEST OVERALL MODEL: {best_overall['model_id']}")
    print(f"   Test MAE: ${best_overall['test_mae']:.2f}")
    print(f"   Test R²: {best_overall['test_r2']:.6f}")
    print(f"   Exact matches: {best_overall['exact_matches']}/{len(test_df)} ({best_overall['exact_matches']/len(test_df)*100:.1f}%)")
    print(f"   Model file: {best_overall['model_file']}")
    print(f"   Scalers file: {best_overall['model_id']}_scalers.pkl")
    
    # Copy the best model to a standard name for easy access
    import shutil
    shutil.copy(best_overall['model_file'], 'best_overall_model.pth')
    shutil.copy(f"{best_overall['model_id']}_scalers.pkl", 'best_overall_scalers.pkl')
    print(f"   ✅ Best model copied to: best_overall_model.pth")
    print(f"   ✅ Best scalers copied to: best_overall_scalers.pkl")
    
    # Save ultra-precise results using the best model
    ultra_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'best_prediction': best_overall['predictions'],
        'error': test_df['reimbursement'] - best_overall['predictions'],
        'abs_error': np.abs(test_df['reimbursement'] - best_overall['predictions'])
    })
    
    ultra_results.to_csv('ultra_deep_results.csv', index=False)
    print(f"\n💾 Ultra-precise results saved to ultra_deep_results.csv")
    
    # Save model comparison results
    comparison_df = pd.DataFrame([
        {
            'model_id': r['model_id'],
            'scaler': r['scaler'], 
            'architecture': r['architecture'],
            'test_mae': r['test_mae'],
            'test_r2': r['test_r2'],
            'exact_matches': r['exact_matches'],
            'close_matches_1': r['close_matches_1'],
            'close_matches_5': r['close_matches_5'],
            'model_file': r['model_file']
        }
        for r in sorted_results
    ])
    comparison_df.to_csv('model_comparison_results.csv', index=False)
    print(f"💾 Model comparison saved to model_comparison_results.csv")
    
    # Analyze the closest predictions from best model
    sorted_by_error = ultra_results.sort_values('abs_error')
    print(f"\n🏆 Top 10 most accurate predictions from best model:")
    for i in range(10):
        row = sorted_by_error.iloc[i]
        print(f"{i+1:2d}. Days: {row['trip_duration_days']:2.0f}, "
              f"Miles: {row['miles_traveled']:4.0f}, "
              f"Receipts: ${row['total_receipts_amount']:7.2f}, "
              f"Actual: ${row['actual_reimbursement']:7.2f}, "
              f"Predicted: ${row['best_prediction']:7.2f}, "
              f"Error: ${row['abs_error']:.4f}")

if __name__ == "__main__":
    main() 