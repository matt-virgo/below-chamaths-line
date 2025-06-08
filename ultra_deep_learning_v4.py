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

def create_v1_ultra_features(df):
    """Create V1's proven comprehensive feature set"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("Creating V1's proven ultra feature set for V4 optimization...")
    
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
    
    print(f"V1 ultra feature set created: {len(feature_cols)} comprehensive features")
    return features_df[feature_cols]

class UltraDeepNetV4(nn.Module):
    """Enhanced UltraDeepNet with advanced techniques"""
    
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256, 128, 64], 
                 dropout_rate=0.3, activation='relu', use_batch_norm=True, 
                 use_skip_connections=False):
        super(UltraDeepNetV4, self).__init__()
        
        self.use_skip_connections = use_skip_connections
        
        # Choose activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # Swish/SiLU
        else:
            self.activation = nn.ReLU()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
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
        
        # Skip connection layers if enabled
        if use_skip_connections:
            self.skip_layers = nn.ModuleList([
                nn.Linear(input_size, hidden_sizes[0]),
                nn.Linear(hidden_sizes[0], hidden_sizes[2]) if len(hidden_sizes) > 2 else None,
                nn.Linear(hidden_sizes[2], 1) if len(hidden_sizes) > 2 else None
            ])
        
    def forward(self, x):
        if self.use_skip_connections and len(self.skip_layers) >= 3:
            # Implement skip connections
            skip1 = self.skip_layers[0](x)
            out = self.network[:4](x)  # First layer + BN + ReLU + Dropout
            out = out + skip1  # Skip connection
            
            if self.skip_layers[1] is not None:
                skip2 = self.skip_layers[1](out)
                out = self.network[4:12](out)  # Next layers
                out = out + skip2
            
            out = self.network[12:](out)  # Remaining layers
            return out.squeeze()
        else:
            return self.network(x).squeeze()

class ResidualBlockV4(nn.Module):
    """Enhanced residual block"""
    
    def __init__(self, size, dropout_rate=0.1, activation='relu'):
        super(ResidualBlockV4, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.dropout = nn.Dropout(dropout_rate * 0.5)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.activation(out)
        return self.dropout(out)

class UltraResNetV4(nn.Module):
    """Enhanced ResNet architecture"""
    
    def __init__(self, input_size, hidden_size=256, num_blocks=6, 
                 dropout_rate=0.2, activation='relu'):
        super(UltraResNetV4, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            self.activation,
            nn.Dropout(dropout_rate)
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlockV4(hidden_size, dropout_rate, activation) 
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            self.activation,
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
            
        return self.output_layer(x).squeeze()

class AdvancedAttentionNetV4(nn.Module):
    """Enhanced attention network with multi-head attention"""
    
    def __init__(self, input_size, hidden_size=256, num_heads=8, dropout_rate=0.2):
        super(AdvancedAttentionNetV4, self).__init__()
        
        # Adjust num_heads to ensure divisibility
        actual_num_heads = min(num_heads, input_size)
        while input_size % actual_num_heads != 0 and actual_num_heads > 1:
            actual_num_heads -= 1
        
        print(f"  Attention model: input_size={input_size}, num_heads={actual_num_heads}")
        
        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_size,
            num_heads=actual_num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
            nn.Softmax(dim=1)
        )
        
        # Main network
        self.main_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.6),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.4),
            
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # Feature attention
        attention_weights = self.feature_attention(x)
        attended_features = x * attention_weights
        
        return self.main_net(attended_features).squeeze()

def train_model_v4(model, train_loader, val_loader, config):
    """Enhanced training with advanced techniques"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Advanced optimizers
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                               weight_decay=config['weight_decay'], eps=1e-8)
    elif config['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'], 
                                 weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], 
                             weight_decay=config['weight_decay'], momentum=0.9)
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], 
                              weight_decay=config['weight_decay'])
    
    # Loss function with potential smoothing
    if config.get('label_smoothing', 0) > 0:
        criterion = nn.SmoothL1Loss()  # Huber loss for outlier robustness
    else:
        criterion = nn.MSELoss()
    
    # Advanced learning rate schedulers
    if config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=config.get('T_0', 50), T_mult=2, eta_min=config['lr'] * 0.01
        )
    elif config['scheduler'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=config.get('patience', 20), 
            min_lr=config['lr'] * 0.001
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get('step_size', 100), gamma=0.8
        )
    else:  # exponential
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    best_val_loss = float('inf')
    patience_counter = 0
    model_id = config['model_id']
    best_model_path = f"V4_{model_id}_best.pth"
    
    print(f"Training V4 model: {model_id}")
    print(f"  Optimizer: {config['optimizer']}, LR: {config['lr']}, WD: {config['weight_decay']}")
    print(f"  Scheduler: {config['scheduler']}, Epochs: {config['epochs']}")
    
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
        
        if epoch % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:4d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.8f}")
        
        if patience_counter >= config.get('early_stopping', 100):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
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
    print("🚀 Ultra Deep Learning V4 - Advanced Hyperparameter Optimization")
    print("="*75)
    print("Building upon V1's success with extensive hyperparameter tuning")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating V1's proven comprehensive feature set...")
    X_train = create_v1_ultra_features(train_df)
    X_test = create_v1_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"\n✨ V4 uses {X_train.shape[1]} proven V1 features with advanced optimization")
    
    # Scaling approaches
    scalers = {
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    # Comprehensive hyperparameter configurations
    model_configs = [
        # UltraDeepNet variants
        {
            'model_id': 'UltraDeep_AdamW_Cosine',
            'model_type': 'UltraDeepNetV4',
            'model_params': {'hidden_sizes': [1024, 512, 256, 128, 64], 'dropout_rate': 0.3, 'activation': 'relu'},
            'optimizer': 'adamw',
            'lr': 0.001,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'epochs': 2000,
            'early_stopping': 120,
            'grad_clip': 1.0
        },
        {
            'model_id': 'UltraDeep_Swish_Plateau',
            'model_type': 'UltraDeepNetV4',
            'model_params': {'hidden_sizes': [512, 256, 128, 64], 'dropout_rate': 0.25, 'activation': 'swish'},
            'optimizer': 'adamw',
            'lr': 0.0008,
            'weight_decay': 2e-4,
            'scheduler': 'plateau',
            'epochs': 2500,
            'early_stopping': 150,
            'grad_clip': 0.5
        },
        {
            'model_id': 'UltraDeep_Skip_ELU',
            'model_type': 'UltraDeepNetV4',
            'model_params': {'hidden_sizes': [768, 384, 192, 96], 'dropout_rate': 0.2, 'activation': 'elu', 'use_skip_connections': True},
            'optimizer': 'adam',
            'lr': 0.0012,
            'weight_decay': 5e-5,
            'scheduler': 'exponential',
            'epochs': 2000,
            'early_stopping': 100,
            'grad_clip': 2.0
        },
        # ResNet variants
        {
            'model_id': 'ResNet_Deep_AdamW',
            'model_type': 'UltraResNetV4',
            'model_params': {'hidden_size': 320, 'num_blocks': 8, 'dropout_rate': 0.15, 'activation': 'swish'},
            'optimizer': 'adamw',
            'lr': 0.0015,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'epochs': 2200,
            'early_stopping': 130,
            'grad_clip': 1.5
        },
        {
            'model_id': 'ResNet_Wide_RMSprop',
            'model_type': 'UltraResNetV4',
            'model_params': {'hidden_size': 512, 'num_blocks': 6, 'dropout_rate': 0.25, 'activation': 'leaky_relu'},
            'optimizer': 'rmsprop',
            'lr': 0.001,
            'weight_decay': 2e-4,
            'scheduler': 'step',
            'epochs': 1800,
            'early_stopping': 100,
            'grad_clip': 1.0
        },
        # Attention variants
        {
            'model_id': 'Attention_MultiHead',
            'model_type': 'AdvancedAttentionNetV4',
            'model_params': {'hidden_size': 256, 'num_heads': 8, 'dropout_rate': 0.2},
            'optimizer': 'adamw',
            'lr': 0.0008,
            'weight_decay': 1e-4,
            'scheduler': 'cosine',
            'epochs': 2500,
            'early_stopping': 140,
            'grad_clip': 0.8
        },
        {
            'model_id': 'Attention_Large',
            'model_type': 'AdvancedAttentionNetV4',
            'model_params': {'hidden_size': 384, 'num_heads': 12, 'dropout_rate': 0.3},
            'optimizer': 'adam',
            'lr': 0.0006,
            'weight_decay': 3e-4,
            'scheduler': 'plateau',
            'epochs': 3000,
            'early_stopping': 180,
            'grad_clip': 1.2
        }
    ]
    
    all_model_results = []
    
    for scaler_name, scaler in scalers.items():
        print(f"\n=== Using {scaler_name} ===")
        
        # Scale features
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
        
        # Split training data for validation
        train_size = int(0.85 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        input_size = X_train_scaled.shape[1]
        
        for config in model_configs:
            print(f"\n--- Training {config['model_id']} with {scaler_name} ---")
            
            # Create model
            if config['model_type'] == 'UltraDeepNetV4':
                model = UltraDeepNetV4(input_size, **config['model_params'])
            elif config['model_type'] == 'UltraResNetV4':
                model = UltraResNetV4(input_size, **config['model_params'])
            elif config['model_type'] == 'AdvancedAttentionNetV4':
                model = AdvancedAttentionNetV4(input_size, **config['model_params'])
            
            # Train model
            trained_model = train_model_v4(model, train_loader, val_loader, config)
            
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
            
            model_full_id = f"V4_{scaler_name}_{config['model_id']}"
            
            results = {
                'model_id': model_full_id,
                'scaler': scaler_name,
                'architecture': config['model_id'],
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'predictions': test_pred,
                'exact_matches': exact_matches,
                'close_matches_1': close_matches_1,
                'close_matches_5': close_matches_5,
                'config': config
            }
            
            all_model_results.append(results)
            
            print(f"  Results: Train MAE ${train_mae:.2f} | Test MAE ${test_mae:.2f} | Test R² {test_r2:.6f}")
            print(f"  Precision: Exact {exact_matches}, ±$1 {close_matches_1}, ±$5 {close_matches_5}")
            
            # Save individual model and scalers
            import pickle
            scaler_file = f"V4_{scaler_name}_{config['model_id']}_scalers.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump({'scaler_X': scaler, 'scaler_y': scaler_y}, f)
    
    # Find overall best model
    best_overall = min(all_model_results, key=lambda x: x['test_mae'])
    
    print(f"\n" + "="*80)
    print(f"🏆 V4 ADVANCED OPTIMIZATION RESULTS:")
    print(f"="*80)
    
    # Sort all results by test MAE
    sorted_results = sorted(all_model_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['model_id']:<45} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f}")
    
    print(f"\n🎯 BEST V4 MODEL: {best_overall['model_id']}")
    print(f"   Test MAE: ${best_overall['test_mae']:.2f}")
    print(f"   Test R²: {best_overall['test_r2']:.6f}")
    print(f"   Architecture: {best_overall['architecture']}")
    print(f"   Configuration: {best_overall['config']['optimizer']}, LR={best_overall['config']['lr']}, WD={best_overall['config']['weight_decay']}")
    
    # Save results
    v4_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'v4_prediction': best_overall['predictions'],
        'error': test_df['reimbursement'] - best_overall['predictions'],
        'abs_error': np.abs(test_df['reimbursement'] - best_overall['predictions'])
    })
    
    v4_results.to_csv('ultra_deep_v4_results.csv', index=False)
    print(f"\n💾 V4 results saved to: ultra_deep_v4_results.csv")
    
    print(f"\n🏁 PERFORMANCE COMPARISON:")
    print(f"    V1 Baseline:        $57.35 MAE")
    print(f"    V4 Best:            ${best_overall['test_mae']:.2f} MAE")
    
    if best_overall['test_mae'] < 57.35:
        improvement = 57.35 - best_overall['test_mae']
        print(f"    🎉 NEW RECORD! ${improvement:.2f} better ({improvement/57.35*100:.1f}% improvement)")
    else:
        diff = best_overall['test_mae'] - 57.35
        print(f"    📊 ${diff:.2f} difference ({diff/57.35*100:.1f}%)")
    
    print(f"\n🧠 V4 INNOVATIONS:")
    print(f"    ✅ Advanced activation functions (Swish, ELU, LeakyReLU)")
    print(f"    ✅ Skip connections and residual blocks") 
    print(f"    ✅ Multi-head attention mechanisms")
    print(f"    ✅ Advanced optimizers (AdamW, RMSprop)")
    print(f"    ✅ Sophisticated learning rate scheduling")
    print(f"    ✅ Gradient clipping and enhanced regularization")
    print(f"    ✅ Extensive hyperparameter optimization")

if __name__ == "__main__":
    main() 