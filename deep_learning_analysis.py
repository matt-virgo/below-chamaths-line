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

def engineer_advanced_features(df):
    """Create an extensive set of features for deep learning"""
    features_df = df.copy()
    
    # Basic derived features
    features_df['miles_per_day'] = features_df['miles_traveled'] / features_df['trip_duration_days']
    features_df['receipts_per_day'] = features_df['total_receipts_amount'] / features_df['trip_duration_days']
    
    # Validated important features from previous analysis
    features_df['total_trip_value'] = (features_df['trip_duration_days'] * 
                                      features_df['miles_traveled'] * 
                                      features_df['total_receipts_amount'])
    
    # Non-linear transformations (these were highly important)
    features_df['receipts_log'] = np.log1p(features_df['total_receipts_amount'])
    features_df['receipts_sqrt'] = np.sqrt(features_df['total_receipts_amount'])
    features_df['receipts_squared'] = features_df['total_receipts_amount'] ** 2
    features_df['receipts_cubed'] = features_df['total_receipts_amount'] ** 3
    
    features_df['miles_log'] = np.log1p(features_df['miles_traveled'])
    features_df['miles_sqrt'] = np.sqrt(features_df['miles_traveled'])
    features_df['miles_squared'] = features_df['miles_traveled'] ** 2
    
    features_df['days_squared'] = features_df['trip_duration_days'] ** 2
    features_df['days_cubed'] = features_df['trip_duration_days'] ** 3
    
    # Lucky cents feature (validated as important)
    features_df['receipts_cents'] = (features_df['total_receipts_amount'] * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(float)
    
    # More complex interactions
    features_df['miles_receipts_interaction'] = features_df['miles_traveled'] * features_df['total_receipts_amount']
    features_df['days_receipts_interaction'] = features_df['trip_duration_days'] * features_df['total_receipts_amount']
    features_df['days_miles_interaction'] = features_df['trip_duration_days'] * features_df['miles_traveled']
    
    # Per-day interactions
    features_df['miles_per_day_squared'] = features_df['miles_per_day'] ** 2
    features_df['receipts_per_day_squared'] = features_df['receipts_per_day'] ** 2
    features_df['miles_receipts_per_day'] = features_df['miles_per_day'] * features_df['receipts_per_day']
    
    # Ratio features
    features_df['receipts_to_miles_ratio'] = features_df['total_receipts_amount'] / (features_df['miles_traveled'] + 1)
    features_df['miles_to_days_ratio'] = features_df['miles_traveled'] / features_df['trip_duration_days']
    
    # Trigonometric features (might capture cyclical patterns)
    features_df['receipts_sin'] = np.sin(features_df['total_receipts_amount'] / 1000)
    features_df['receipts_cos'] = np.cos(features_df['total_receipts_amount'] / 1000)
    features_df['miles_sin'] = np.sin(features_df['miles_traveled'] / 500)
    features_df['miles_cos'] = np.cos(features_df['miles_traveled'] / 500)
    
    # Exponential features
    features_df['receipts_exp'] = np.exp(features_df['total_receipts_amount'] / 2000)
    features_df['miles_exp'] = np.exp(features_df['miles_traveled'] / 1000)
    
    # Binning features
    features_df['receipts_bin'] = pd.cut(features_df['total_receipts_amount'], bins=10, labels=False)
    features_df['miles_bin'] = pd.cut(features_df['miles_traveled'], bins=10, labels=False)
    features_df['days_bin'] = pd.cut(features_df['trip_duration_days'], bins=5, labels=False)
    
    # Remove the target if it exists
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    return features_df[feature_cols]

class DeepReimbursementNet(nn.Module):
    """Deep neural network for reimbursement prediction"""
    
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64, 32], dropout_rate=0.3):
        super(DeepReimbursementNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class WideDeepNet(nn.Module):
    """Wide & Deep model architecture"""
    
    def __init__(self, input_size, wide_features=3):
        super(WideDeepNet, self).__init__()
        
        # Wide part (linear)
        self.wide = nn.Linear(wide_features, 1)
        
        # Deep part
        self.deep = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Split input for wide and deep parts
        wide_input = x[:, :3]  # First 3 features for wide part
        
        wide_out = self.wide(wide_input)
        deep_out = self.deep(x)
        
        return (wide_out + deep_out).squeeze()

class ResidualBlock(nn.Module):
    """Residual block for deep networks"""
    
    def __init__(self, size):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    """Residual Network for tabular data"""
    
    def __init__(self, input_size, hidden_size=128, num_blocks=4):
        super(ResNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_blocks)])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

def train_model(model, train_loader, val_loader, epochs=1000, lr=0.001, patience=50):
    """Train a neural network model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5)
    
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
                val_loss += criterion(outputs, batch_y).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
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
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Engineering advanced features...")
    X_train = engineer_advanced_features(train_df)
    X_test = engineer_advanced_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Scale features
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target (helps with training stability)
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
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_size = X_train_scaled.shape[1]
    
    # Test different architectures
    models = {
        'Deep Network': DeepReimbursementNet(input_size),
        'Wide & Deep': WideDeepNet(input_size),
        'ResNet': ResNet(input_size)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        trained_model = train_model(model, train_loader, val_loader, epochs=1000)
        
        # Evaluate
        train_pred, train_actual, train_mae, train_rmse, train_r2 = evaluate_model(
            trained_model, train_loader, scaler_y
        )
        test_pred, test_actual, test_mae, test_rmse, test_r2 = evaluate_model(
            trained_model, test_loader, scaler_y
        )
        
        results[name] = {
            'model': trained_model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': test_pred
        }
        
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Test MAE:  ${test_mae:.2f}")
        print(f"  Train R²:  {train_r2:.6f}")
        print(f"  Test R²:   {test_r2:.6f}")
        
        # Check for near-perfect predictions
        exact_matches = np.sum(np.abs(test_actual - test_pred) < 0.01)
        close_matches = np.sum(np.abs(test_actual - test_pred) < 1.0)
        
        print(f"  Exact matches (±$0.01): {exact_matches}/{len(test_actual)} ({exact_matches/len(test_actual)*100:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches}/{len(test_actual)} ({close_matches/len(test_actual)*100:.1f}%)")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_model_results = results[best_model_name]
    
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Test MAE: ${best_model_results['test_mae']:.2f}")
    print(f"Test R²: {best_model_results['test_r2']:.6f}")
    
    # Save detailed predictions
    predictions_df = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement']
    })
    
    for name, model_results in results.items():
        predictions_df[f'{name}_prediction'] = model_results['predictions']
        predictions_df[f'{name}_error'] = test_df['reimbursement'] - model_results['predictions']
    
    predictions_df.to_csv('deep_learning_predictions.csv', index=False)
    print("Detailed predictions saved to deep_learning_predictions.csv")

if __name__ == "__main__":
    main() 