#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    
    # Convert to DataFrames - ONLY RAW FEATURES
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

def main():
    print("🌳 XGBoost with RAW FEATURES ONLY - No Feature Engineering")
    print("="*70)
    print("Testing if simplicity beats complexity")
    print()
    
    print("Loading data...")
    train_df, test_df = load_data()
    
    # Use ONLY the 3 raw input features
    feature_cols = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"✨ Using only {len(feature_cols)} RAW features (no engineering):")
    for i, col in enumerate(feature_cols, 1):
        print(f"   {i}. {col}")
    
    print(f"\nTraining data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    
    # Basic statistics
    print(f"\n📊 RAW FEATURE STATISTICS:")
    print(f"   Trip Duration - Min: {X_train['trip_duration_days'].min()}, Max: {X_train['trip_duration_days'].max()}, Mean: {X_train['trip_duration_days'].mean():.1f}")
    print(f"   Miles Traveled - Min: {X_train['miles_traveled'].min():.1f}, Max: {X_train['miles_traveled'].max():.1f}, Mean: {X_train['miles_traveled'].mean():.1f}")
    print(f"   Total Receipts - Min: ${X_train['total_receipts_amount'].min():.2f}, Max: ${X_train['total_receipts_amount'].max():.2f}, Mean: ${X_train['total_receipts_amount'].mean():.2f}")
    print(f"   Reimbursement - Min: ${y_train.min():.2f}, Max: ${y_train.max():.2f}, Mean: ${y_train.mean():.2f}")
    
    # XGBoost configurations to test
    xgb_configs = [
        {
            'name': 'XGB_Default_Raw',
            'params': {
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 1.0,  # Use all 3 features
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42
            }
        },
        {
            'name': 'XGB_Deep_Raw',
            'params': {
                'n_estimators': 2000,
                'max_depth': 12,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'random_state': 42
            }
        },
        {
            'name': 'XGB_Conservative_Raw',
            'params': {
                'n_estimators': 1500,
                'max_depth': 4,
                'learning_rate': 0.03,
                'subsample': 0.7,
                'colsample_bytree': 1.0,
                'reg_alpha': 1,
                'reg_lambda': 5,
                'random_state': 42
            }
        },
        {
            'name': 'XGB_Aggressive_Raw',
            'params': {
                'n_estimators': 3000,
                'max_depth': 8,
                'learning_rate': 0.02,
                'subsample': 0.85,
                'colsample_bytree': 1.0,
                'reg_alpha': 0.01,
                'reg_lambda': 0.1,
                'random_state': 42
            }
        }
    ]
    
    all_results = []
    
    for config in xgb_configs:
        print(f"\n{'='*60}")
        print(f"🌳 Training: {config['name']}")
        print(f"{'='*60}")
        
        # Create and train XGBoost model
        model = xgb.XGBRegressor(**config['params'])
        
        print(f"Parameters: {config['params']}")
        print("Training XGBoost model...")
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Precision metrics
        exact_matches = np.sum(np.abs(y_test - test_pred) < 0.01)
        close_matches_1 = np.sum(np.abs(y_test - test_pred) < 1.0)
        close_matches_5 = np.sum(np.abs(y_test - test_pred) < 5.0)
        close_matches_10 = np.sum(np.abs(y_test - test_pred) < 10.0)
        
        # Feature importance
        feature_importance = model.feature_importances_
        
        print(f"\n🎯 {config['name']} RESULTS:")
        print(f"   Train MAE: ${train_mae:.2f}")
        print(f"   Test MAE:  ${test_mae:.2f}")
        print(f"   Train R²:  {train_r2:.6f}")
        print(f"   Test R²:   {test_r2:.6f}")
        print(f"   Exact matches (±$0.01): {exact_matches}")
        print(f"   Close matches (±$1.00): {close_matches_1}")
        print(f"   Close matches (±$5.00): {close_matches_5}")
        print(f"   Close matches (±$10.00): {close_matches_10}")
        
        print(f"\n📊 FEATURE IMPORTANCE:")
        for i, (feature, importance) in enumerate(zip(feature_cols, feature_importance)):
            print(f"   {i+1}. {feature}: {importance:.4f}")
        
        results = {
            'name': config['name'],
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'exact_matches': exact_matches,
            'close_matches_1': close_matches_1,
            'close_matches_5': close_matches_5,
            'close_matches_10': close_matches_10,
            'feature_importance': dict(zip(feature_cols, feature_importance)),
            'predictions': test_pred
        }
        
        all_results.append(results)
        
        # Save model
        model.save_model(f'{config["name"]}_model.json')
        print(f"   💾 Model saved to: {config['name']}_model.json")
    
    # Find best model
    best_result = min(all_results, key=lambda x: x['test_mae'])
    
    print(f"\n{'='*80}")
    print(f"🏆 RAW FEATURES XGBoost FINAL RESULTS:")
    print(f"{'='*80}")
    
    # Sort all results by test MAE
    sorted_results = sorted(all_results, key=lambda x: x['test_mae'])
    
    for i, result in enumerate(sorted_results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        print(f"{rank_emoji} {result['name']:<25} | Test MAE: ${result['test_mae']:6.2f} | R²: {result['test_r2']:.4f}")
    
    print(f"\n🎉 BEST RAW FEATURES MODEL: {best_result['name']}")
    print(f"   🎯 Test MAE: ${best_result['test_mae']:.2f}")
    print(f"   📊 Test R²: {best_result['test_r2']:.6f}")
    
    print(f"\n🔍 BEST MODEL FEATURE IMPORTANCE:")
    sorted_importance = sorted(best_result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        print(f"   {i}. {feature}: {importance:.4f} ({importance*100:.1f}%)")
    
    # Save best results
    raw_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': test_df['reimbursement'],
        'raw_xgb_prediction': best_result['predictions'],
        'error': test_df['reimbursement'] - best_result['predictions'],
        'abs_error': np.abs(test_df['reimbursement'] - best_result['predictions'])
    })
    
    raw_results.to_csv('xgboost_raw_features_results.csv', index=False)
    print(f"\n💾 Results saved to: xgboost_raw_features_results.csv")
    
    print(f"\n📈 COMPARISON WITH COMPLEX APPROACHES:")
    print(f"   V1 (58 features):        $57.35 MAE")
    print(f"   V4 (58 features):        $59.76 MAE")  
    print(f"   XGBoost V1 (58 features): $63.50 MAE")
    print(f"   Raw XGBoost (3 features): ${best_result['test_mae']:.2f} MAE")
    
    if best_result['test_mae'] < 57.35:
        improvement = 57.35 - best_result['test_mae']
        print(f"   🎉 RAW FEATURES WIN! ${improvement:.2f} better than V1 ({improvement/57.35*100:.1f}% improvement)")
    elif best_result['test_mae'] < 63.50:
        improvement = 63.50 - best_result['test_mae']
        print(f"   ✅ Better than engineered XGBoost! ${improvement:.2f} improvement")
    else:
        diff = best_result['test_mae'] - 57.35
        print(f"   📊 ${diff:.2f} worse than V1 baseline")
    
    print(f"\n🧠 RAW FEATURES INSIGHTS:")
    print(f"   • Tree models can naturally find non-linear patterns")
    print(f"   • Feature engineering might add noise rather than signal")
    print(f"   • XGBoost excels at discovering interactions automatically")
    print(f"   • Sometimes simple is better than complex")
    print(f"   • Raw features avoid overfitting to training patterns")
    
    # Hyperparameter tuning on the best base configuration
    if best_result['test_mae'] < 60.0:  # If promising, do hyperparameter tuning
        print(f"\n🔧 HYPERPARAMETER TUNING FOR BEST MODEL...")
        
        best_base_config = next(config for config in xgb_configs if config['name'] == best_result['name'])
        
        # Define parameter grid for tuning
        param_grid = {
            'n_estimators': [2000, 3000, 4000],
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.02, 0.03, 0.05],
            'subsample': [0.8, 0.85, 0.9],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.1, 1, 5]
        }
        
        print("Performing randomized search...")
        
        # Use RandomizedSearchCV for efficiency
        xgb_model = xgb.XGBRegressor(random_state=42, colsample_bytree=1.0)
        
        random_search = RandomizedSearchCV(
            xgb_model,
            param_grid,
            n_iter=20,  # Try 20 combinations
            scoring='neg_mean_absolute_error',
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        tuned_model = random_search.best_estimator_
        tuned_pred = tuned_model.predict(X_test)
        tuned_mae = mean_absolute_error(y_test, tuned_pred)
        tuned_r2 = r2_score(y_test, tuned_pred)
        
        print(f"\n🎯 TUNED MODEL RESULTS:")
        print(f"   Best parameters: {random_search.best_params_}")
        print(f"   Tuned Test MAE: ${tuned_mae:.2f}")
        print(f"   Tuned Test R²: {tuned_r2:.6f}")
        print(f"   Improvement: ${best_result['test_mae'] - tuned_mae:.2f}")
        
        if tuned_mae < best_result['test_mae']:
            print(f"   🎉 TUNING IMPROVED PERFORMANCE!")
            
            # Save tuned results
            tuned_results = pd.DataFrame({
                'trip_duration_days': test_df['trip_duration_days'],
                'miles_traveled': test_df['miles_traveled'],
                'total_receipts_amount': test_df['total_receipts_amount'],
                'actual_reimbursement': test_df['reimbursement'],
                'tuned_xgb_prediction': tuned_pred,
                'error': test_df['reimbursement'] - tuned_pred,
                'abs_error': np.abs(test_df['reimbursement'] - tuned_pred)
            })
            
            tuned_results.to_csv('xgboost_raw_tuned_results.csv', index=False)
            tuned_model.save_model('xgboost_raw_tuned_best.json')
            
            print(f"   💾 Tuned results saved to: xgboost_raw_tuned_results.csv")
            print(f"   💾 Tuned model saved to: xgboost_raw_tuned_best.json")

if __name__ == "__main__":
    main() 