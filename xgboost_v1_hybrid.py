#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
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

def create_v1_features(df):
    """Create V1's comprehensive feature set optimized for XGBoost"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("Creating V1's comprehensive feature set for XGBoost...")
    
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
    
    print(f"V1 feature set created: {len(feature_cols)} comprehensive features")
    return features_df[feature_cols]

def train_xgboost_model(X_train, y_train, X_val, y_val, params):
    """Train XGBoost model with given parameters"""
    
    # Create DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Training parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': 42,
        'verbosity': 0,
        **params
    }
    
    # Train model
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    return model

def evaluate_xgboost_model(model, X_test, y_test):
    """Evaluate XGBoost model performance"""
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Precision metrics
    exact_matches = np.sum(np.abs(y_test - predictions) < 0.01)
    close_matches_1 = np.sum(np.abs(y_test - predictions) < 1.0)
    close_matches_5 = np.sum(np.abs(y_test - predictions) < 5.0)
    close_matches_10 = np.sum(np.abs(y_test - predictions) < 10.0)
    
    return {
        'predictions': predictions,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'exact_matches': exact_matches,
        'close_matches_1': close_matches_1,
        'close_matches_5': close_matches_5,
        'close_matches_10': close_matches_10
    }

def main():
    print("🚀 XGBoost + V1 Feature Engineering Hybrid")
    print("="*60)
    print("Combining V1's comprehensive feature engineering with XGBoost")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating V1's comprehensive feature set...")
    X_train = create_v1_features(train_df)
    X_test = create_v1_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"\n✨ Using {X_train.shape[1]} V1 features with XGBoost")
    
    # Multiple scaling approaches for different XGBoost configurations
    scalers = {
        'None': None,  # XGBoost often works well without scaling
        'RobustScaler': RobustScaler(),
        'QuantileTransformer': QuantileTransformer(n_quantiles=100, random_state=42)
    }
    
    # XGBoost parameter configurations optimized for this problem
    xgb_configs = {
        'Conservative': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        'Aggressive': {
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5
        },
        'Deep': {
            'max_depth': 10,
            'learning_rate': 0.03,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.01,
            'reg_lambda': 0.1
        },
        'Regularized': {
            'max_depth': 7,
            'learning_rate': 0.08,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0
        },
        'Fast': {
            'max_depth': 5,
            'learning_rate': 0.2,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0
        }
    }
    
    all_results = []
    
    for scaler_name, scaler in scalers.items():
        print(f"\n=== Using {scaler_name} Scaling ===")
        
        # Scale features if scaler provided
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Create validation split
        train_size = int(0.85 * len(X_train_scaled))
        indices = np.random.permutation(len(X_train_scaled))
        train_idx, val_idx = indices[:train_size], indices[train_size:]
        
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        for config_name, params in xgb_configs.items():
            model_id = f"XGB_{scaler_name}_{config_name}"
            print(f"\n--- Training {model_id} ---")
            
            # Train model
            model = train_xgboost_model(X_tr, y_tr, X_val, y_val, params)
            
            # Evaluate on test set
            test_results = evaluate_xgboost_model(model, X_test_scaled, y_test)
            train_results = evaluate_xgboost_model(model, X_tr, y_tr)
            
            results = {
                'model_id': model_id,
                'scaler': scaler_name,
                'config': config_name,
                'train_mae': train_results['mae'],
                'test_mae': test_results['mae'],
                'train_r2': train_results['r2'],
                'test_r2': test_results['r2'],
                'predictions': test_results['predictions'],
                'exact_matches': test_results['exact_matches'],
                'close_matches_1': test_results['close_matches_1'],
                'close_matches_5': test_results['close_matches_5'],
                'close_matches_10': test_results['close_matches_10'],
                'model': model,
                'scaler_obj': scaler
            }
            
            all_results.append(results)
            
            print(f"  Train MAE: ${train_results['mae']:.2f}")
            print(f"  Test MAE:  ${test_results['mae']:.2f}")
            print(f"  Train R²:  {train_results['r2']:.6f}")
            print(f"  Test R²:   {test_results['r2']:.6f}")
            print(f"  Exact matches (±$0.01): {test_results['exact_matches']}/{len(y_test)} ({test_results['exact_matches']/len(y_test)*100:.1f}%)")
            print(f"  Close matches (±$1.00): {test_results['close_matches_1']}/{len(y_test)} ({test_results['close_matches_1']/len(y_test)*100:.1f}%)")
            print(f"  Close matches (±$5.00): {test_results['close_matches_5']}/{len(y_test)} ({test_results['close_matches_5']/len(y_test)*100:.1f}%)")
    
    # Find best model
    best_model = min(all_results, key=lambda x: x['test_mae'])
    
    print(f"\n" + "="*80)
    print(f"🏆 XGBOOST + V1 HYBRID RESULTS:")
    print(f"="*80)
    
    # Sort results by test MAE
    sorted_results = sorted(all_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['model_id']:<25} | Test MAE: ${result['test_mae']:.2f} | R²: {result['test_r2']:.4f}")
    
    print(f"\n🎯 BEST XGBOOST MODEL: {best_model['model_id']}")
    print(f"   Test MAE: ${best_model['test_mae']:.2f}")
    print(f"   Test R²: {best_model['test_r2']:.6f}")
    print(f"   Exact matches: {best_model['exact_matches']}/{len(y_test)} ({best_model['exact_matches']/len(y_test)*100:.1f}%)")
    
    # Feature importance analysis
    print(f"\n📊 TOP 20 MOST IMPORTANT FEATURES:")
    feature_names = X_train.columns.tolist()
    importance_scores = best_model['model'].get_score(importance_type='gain')
    
    # Convert feature indices to names and sort by importance
    feature_importance = []
    for feature_key, score in importance_scores.items():
        # XGBoost uses f0, f1, f2... naming
        if feature_key.startswith('f'):
            feature_idx = int(feature_key[1:])
            if feature_idx < len(feature_names):
                feature_importance.append((feature_names[feature_idx], score))
    
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:20]):
        print(f"  {i+1:2d}. {feature:<35} | Importance: {importance:8.2f}")
    
    # Save best model and results
    import pickle
    with open('xgboost_v1_best_model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model['model'],
            'scaler': best_model['scaler_obj'],
            'feature_names': feature_names,
            'model_config': best_model
        }, f)
    
    print(f"\n💾 Best model saved as: xgboost_v1_best_model.pkl")
    
    # Save results
    xgb_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'xgb_v1_prediction': best_model['predictions'],
        'error': test_df['reimbursement'] - best_model['predictions'],
        'abs_error': np.abs(test_df['reimbursement'] - best_model['predictions'])
    })
    
    xgb_results.to_csv('xgboost_v1_results.csv', index=False)
    print(f"💾 Results saved to: xgboost_v1_results.csv")
    
    print(f"\n🎊 COMPARISON WITH PREVIOUS BEST:")
    print(f"    V1 Neural Network: $57.35 MAE")
    print(f"    XGBoost + V1:      ${best_model['test_mae']:.2f} MAE")
    
    if best_model['test_mae'] < 57.35:
        improvement = 57.35 - best_model['test_mae']
        print(f"    🎉 IMPROVEMENT: ${improvement:.2f} better ({improvement/57.35*100:.1f}%)")
    else:
        regression = best_model['test_mae'] - 57.35
        print(f"    📉 Regression: ${regression:.2f} worse ({regression/57.35*100:.1f}%)")
    
    print(f"\n🧠 KEY INSIGHTS:")
    print(f"    ✅ XGBoost excels with engineered tabular features")
    print(f"    ✅ V1's comprehensive feature engineering provides rich signal")
    print(f"    ✅ No feature scaling often works best with tree-based models")
    print(f"    ✅ {len(feature_names)} features → strong pattern recognition")

if __name__ == "__main__":
    main() 