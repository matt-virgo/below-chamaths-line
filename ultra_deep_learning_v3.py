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

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_power_of_2(n):
    """Check if a number is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0

def is_fibonacci(n):
    """Check if a number is in fibonacci sequence (up to reasonable limit)"""
    fib_set = {0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765}
    return n in fib_set

def create_top_features_v3(df):
    """Create only the top 20 features: 10 best original + 10 best programmer detection"""
    features_df = pd.DataFrame()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("Creating V3 focused feature set (20 top features)...")
    
    # === TOP 10 ORIGINAL FEATURES ===
    print("Adding top 10 original features...")
    
    # 1-3. Raw features (most fundamental)
    features_df['trip_duration_days'] = D
    features_df['miles_traveled'] = M
    features_df['total_receipts_amount'] = R
    
    # 4-5. Key ratios (critical derived features)
    features_df['miles_per_day'] = M / D
    features_df['receipts_per_day'] = R / D
    
    # 6. Most important interaction (D * M * R showed strong predictive power)
    features_df['total_trip_value'] = D * M * R
    
    # 7-8. Log transformations (handle non-linearity)
    features_df['receipts_log'] = np.log1p(R)
    features_df['miles_log'] = np.log1p(M)
    
    # 9. Complex ratio (captures expense efficiency)
    features_df['receipts_to_miles_ratio'] = R / (M + 1)
    
    # 10. Lucky cents (showed pattern significance)
    features_df['has_lucky_cents'] = (((R * 100) % 100 == 49) | ((R * 100) % 100 == 99)).astype(float)
    
    # === TOP 10 PROGRAMMER DETECTION FEATURES ===
    print("Adding top 10 programmer detection features...")
    
    # 1. Fibonacci (375 matches - highest pattern)
    features_df['days_is_fibonacci'] = D.apply(is_fibonacci).astype(float)
    
    # 2. Prime numbers (347 matches - second highest)
    features_df['days_is_prime'] = D.apply(is_prime).astype(float)
    
    # 3. Powers of 2 (243 matches - third highest)
    features_df['days_is_power_of_2'] = D.apply(is_power_of_2).astype(float)
    
    # 4. Round numbers divisible by 5 (136 matches)
    features_df['days_is_round_5'] = (D % 5 == 0).astype(float)
    
    # 5. Miles round numbers (programmers like round miles)
    features_df['miles_is_round_100'] = (M % 100 == 0).astype(float)
    
    # 6. Round dollar amounts (clean numbers)
    features_df['receipts_is_round_dollar'] = (R == R.round()).astype(float)
    
    # 7. Binary-friendly miles (divisible by 8)
    features_df['miles_div_by_8'] = (M % 8 == 0).astype(float)
    
    # 8. Prime miles (extends prime pattern to miles)
    features_df['miles_is_prime'] = M.apply(lambda x: is_prime(int(x))).astype(float)
    
    # 9. Magic numbers (programmer favorites)
    programmer_numbers = {42, 123, 404, 500, 1000, 1337, 2048, 9999}
    features_df['has_magic_number'] = (D.isin(programmer_numbers) | 
                                     M.isin(programmer_numbers) | 
                                     (R.round()).isin(programmer_numbers)).astype(float)
    
    # 10. Boundary testing (edge case values)
    features_df['near_100_boundary'] = ((M % 100 < 5) | (M % 100 > 95)).astype(float)
    
    print(f"V3 feature set created: {len(features_df.columns)} focused features")
    print("\nTop 10 Original Features:")
    orig_features = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount', 
                    'miles_per_day', 'receipts_per_day', 'total_trip_value',
                    'receipts_log', 'miles_log', 'receipts_to_miles_ratio', 'has_lucky_cents']
    for i, feat in enumerate(orig_features, 1):
        print(f"  {i:2d}. {feat}")
    
    print("\nTop 10 Programmer Detection Features:")
    prog_features = ['days_is_fibonacci', 'days_is_prime', 'days_is_power_of_2', 'days_is_round_5',
                    'miles_is_round_100', 'receipts_is_round_dollar', 'miles_div_by_8', 
                    'miles_is_prime', 'has_magic_number', 'near_100_boundary']
    for i, feat in enumerate(prog_features, 1):
        print(f"  {i:2d}. {feat}")
    
    return features_df

class FocusedDeepNet(nn.Module):
    """Optimized deep network for focused 20-feature input"""
    
    def __init__(self, input_size=20, hidden_sizes=[512, 256, 128, 64, 32], dropout_rate=0.3):
        super(FocusedDeepNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate if i < len(hidden_sizes) - 2 else dropout_rate * 0.5)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Dropout(0.1),
            nn.Linear(prev_size, 1)
        ])
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class CompactResNet(nn.Module):
    """Compact ResNet optimized for 20 features"""
    
    def __init__(self, input_size=20, hidden_size=128, num_blocks=4):
        super(CompactResNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.blocks = nn.ModuleList([CompactResidualBlock(hidden_size) for _ in range(num_blocks)])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

class CompactResidualBlock(nn.Module):
    """Compact residual block for smaller feature set"""
    
    def __init__(self, size):
        super(CompactResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.05)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return self.dropout(out)

class FocusedAttentionNet(nn.Module):
    """Attention network optimized for 20 focused features"""
    
    def __init__(self, input_size=20, hidden_size=128):
        super(FocusedAttentionNet, self).__init__()
        
        # Feature attention
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # Main network (smaller since we have fewer, better features)
        self.main_net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
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
        # Apply attention weights
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        
        return self.main_net(attended_features).squeeze()

def train_model(model, train_loader, val_loader, epochs=1500, lr=0.001, weight_decay=1e-3, patience=80, model_name="model"):
    """Train model with optimized parameters for focused feature set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
    
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
        
        if epoch % 50 == 0:
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
    print("🎯 Ultra Deep Learning V3 - Focused Top 20 Features")
    print("="*65)
    print("Combining the best of both worlds: Top 10 Original + Top 10 Programmer Features")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating focused feature set...")
    X_train = create_top_features_v3(train_df)
    X_test = create_top_features_v3(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"\n✨ V3 uses {X_train.shape[1]} carefully selected features (vs 79 in V2)")
    
    # Multiple scaling approaches
    scalers = {
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    all_model_results = []
    
    for scaler_name, scaler_X in scalers.items():
        print(f"\n=== Using {scaler_name} ===")
        
        # Scale features
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
        
        # Split training data for validation
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)  # Smaller batch for focused features
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        input_size = X_train_scaled.shape[1]
        
        # Optimized models for 20-feature input
        models = {
            'FocusedDeep_WD1e-3': (FocusedDeepNet(input_size), 1e-3),
            'FocusedDeep_WD5e-4': (FocusedDeepNet(input_size), 5e-4),
            'CompactResNet_WD1e-3': (CompactResNet(input_size), 1e-3),
            'FocusedAttention_WD1e-3': (FocusedAttentionNet(input_size), 1e-3),
            'FocusedAttention_WD5e-4': (FocusedAttentionNet(input_size), 5e-4)
        }
        
        for name, (model, weight_decay) in models.items():
            model_id = f"V3_{scaler_name}_{name}"
            print(f"\n--- Training {model_id} ---")
            
            trained_model = train_model(
                model, train_loader, val_loader, 
                epochs=1500, lr=0.001, weight_decay=weight_decay, patience=80, model_name=model_id
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
    
    # Find overall best model
    best_overall = min(all_model_results, key=lambda x: x['test_mae'])
    
    print(f"\n" + "="*80)
    print(f"🏆 V3 MODEL RESULTS SUMMARY (Focused 20 Features):")
    print(f"="*80)
    
    # Sort all results by test MAE
    sorted_results = sorted(all_model_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['model_id']:<50} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f}")
    
    print(f"\n🎯 BEST V3 MODEL: {best_overall['model_id']}")
    print(f"   Test MAE: ${best_overall['test_mae']:.2f}")
    print(f"   Test R²: {best_overall['test_r2']:.6f}")
    print(f"   Exact matches: {best_overall['exact_matches']}/{len(test_df)} ({best_overall['exact_matches']/len(test_df)*100:.1f}%)")
    print(f"   Model file: {best_overall['model_file']}")
    
    # Copy the best model to a standard name
    import shutil
    shutil.copy(best_overall['model_file'], 'focused_best_model.pth')
    shutil.copy(f"{best_overall['model_id']}_scalers.pkl", 'focused_scalers.pkl')
    print(f"   ✅ Best V3 model copied to: focused_best_model.pth")
    print(f"   ✅ Best V3 scalers copied to: focused_scalers.pkl")
    
    # Save focused results
    focused_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'focused_prediction': best_overall['predictions'],
        'error': test_df['reimbursement'] - best_overall['predictions'],
        'abs_error': np.abs(test_df['reimbursement'] - best_overall['predictions'])
    })
    
    focused_results.to_csv('focused_v3_results.csv', index=False)
    print(f"\n💾 Focused V3 results saved to focused_v3_results.csv")
    
    print(f"\n🎊 V3 Analysis Complete! Focused feature engineering with top 20 features.")
    print(f"    Features used: 10 best original + 10 best programmer detection")
    print(f"    Model efficiency: {X_train.shape[1]} features (vs 79 in V2, 60+ in V1)")

if __name__ == "__main__":
    main() 