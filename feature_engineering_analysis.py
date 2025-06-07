#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

def engineer_features(df):
    """Create features based on employee interview insights"""
    
    # Copy the dataframe
    features_df = df.copy()
    
    # Basic derived features
    features_df['miles_per_day'] = features_df['miles_traveled'] / features_df['trip_duration_days']
    features_df['receipts_per_day'] = features_df['total_receipts_amount'] / features_df['trip_duration_days']
    
    # Per diem assumptions (Lisa: ~$100/day base)
    features_df['base_per_diem'] = features_df['trip_duration_days'] * 100
    
    # Trip length categories (from interviews)
    features_df['is_short_trip'] = (features_df['trip_duration_days'] <= 2).astype(int)
    features_df['is_medium_trip'] = ((features_df['trip_duration_days'] >= 3) & 
                                    (features_df['trip_duration_days'] <= 6)).astype(int)
    features_df['is_long_trip'] = (features_df['trip_duration_days'] >= 7).astype(int)
    
    # 5-day trip bonus (multiple employees mentioned this)
    features_df['is_5_day_trip'] = (features_df['trip_duration_days'] == 5).astype(int)
    
    # Vacation penalty (Kevin: 8+ days with high spending)
    features_df['is_vacation_penalty'] = ((features_df['trip_duration_days'] >= 8) & 
                                         (features_df['receipts_per_day'] > 120)).astype(int)
    
    # Efficiency ranges (Kevin's sweet spot: 180-220 miles/day)
    features_df['is_optimal_efficiency'] = ((features_df['miles_per_day'] >= 180) & 
                                           (features_df['miles_per_day'] <= 220)).astype(int)
    features_df['is_low_efficiency'] = (features_df['miles_per_day'] < 100).astype(int)
    features_df['is_high_efficiency'] = (features_df['miles_per_day'] > 300).astype(int)
    
    # Kevin's sweet spot combo: 5-day + 180+ miles/day + <$100/day spending
    features_df['is_sweet_spot_combo'] = ((features_df['trip_duration_days'] == 5) & 
                                         (features_df['miles_per_day'] >= 180) & 
                                         (features_df['receipts_per_day'] < 100)).astype(int)
    
    # Receipt amount thresholds (from interviews)
    features_df['is_tiny_receipts'] = (features_df['total_receipts_amount'] < 50).astype(int)
    features_df['is_low_receipts'] = ((features_df['total_receipts_amount'] >= 50) & 
                                     (features_df['total_receipts_amount'] < 200)).astype(int)
    features_df['is_medium_receipts'] = ((features_df['total_receipts_amount'] >= 200) & 
                                        (features_df['total_receipts_amount'] < 800)).astype(int)
    features_df['is_high_receipts'] = (features_df['total_receipts_amount'] >= 800).astype(int)
    
    # Spending per day categories (Kevin's optimal ranges)
    features_df['spending_category'] = 0  # Default
    features_df.loc[features_df['receipts_per_day'] < 75, 'spending_category'] = 1  # Low
    features_df.loc[(features_df['receipts_per_day'] >= 75) & 
                   (features_df['receipts_per_day'] <= 120), 'spending_category'] = 2  # Optimal
    features_df.loc[features_df['receipts_per_day'] > 120, 'spending_category'] = 3  # High
    
    # Mileage tiers (Lisa: different rates after 100 miles)
    features_df['mileage_tier_1'] = np.minimum(features_df['miles_traveled'], 100)
    features_df['mileage_tier_2'] = np.maximum(0, np.minimum(features_df['miles_traveled'] - 100, 400))
    features_df['mileage_tier_3'] = np.maximum(0, features_df['miles_traveled'] - 500)
    
    # Potential rounding quirks (Lisa: .49 and .99 cents)
    features_df['receipts_cents'] = (features_df['total_receipts_amount'] * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(int)
    
    # Interaction features (Kevin emphasized these)
    features_df['trip_efficiency_interaction'] = (features_df['trip_duration_days'] * 
                                                 features_df['miles_per_day'])
    features_df['spending_mileage_interaction'] = (features_df['receipts_per_day'] * 
                                                  features_df['miles_traveled'])
    features_df['total_trip_value'] = (features_df['trip_duration_days'] * 
                                      features_df['miles_traveled'] * 
                                      features_df['total_receipts_amount'])
    
    # Non-linear transformations
    features_df['miles_log'] = np.log1p(features_df['miles_traveled'])
    features_df['receipts_log'] = np.log1p(features_df['total_receipts_amount'])
    features_df['miles_sqrt'] = np.sqrt(features_df['miles_traveled'])
    features_df['receipts_sqrt'] = np.sqrt(features_df['total_receipts_amount'])
    
    # Polynomial features for potential curves
    features_df['miles_squared'] = features_df['miles_traveled'] ** 2
    features_df['days_squared'] = features_df['trip_duration_days'] ** 2
    features_df['receipts_squared'] = features_df['total_receipts_amount'] ** 2
    
    return features_df

def analyze_feature_importance(X_train, y_train, feature_names):
    """Analyze which features are most important using multiple methods"""
    
    print("=== FEATURE IMPORTANCE ANALYSIS ===\n")
    
    # Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Gradient Boosting for feature importance
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf.feature_importances_,
        'gb_importance': gb.feature_importances_
    })
    
    # Average importance
    importance_df['avg_importance'] = (importance_df['rf_importance'] + 
                                      importance_df['gb_importance']) / 2
    
    # Sort by average importance
    importance_df = importance_df.sort_values('avg_importance', ascending=False)
    
    print("Top 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    print()
    
    # Statistical correlations
    correlations = []
    for feature in feature_names:
        if feature in X_train.columns:
            corr = np.corrcoef(X_train[feature], y_train)[0, 1]
            if not np.isnan(corr):
                correlations.append((feature, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 Features by Correlation with Reimbursement:")
    for feature, corr in correlations[:10]:
        print(f"{feature:30s}: {corr:.4f}")
    print()
    
    return importance_df

def test_models(X_train, y_train, X_test, y_test):
    """Test multiple model types and return their performance"""
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    print("=== MODEL PERFORMANCE COMPARISON ===\n")
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        results[name] = {
            'model': model,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': test_pred
        }
        
        print(f"{name}:")
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Test MAE:  ${test_mae:.2f}")
        print(f"  Train R²:  {train_r2:.4f}")
        print(f"  Test R²:   {test_r2:.4f}")
        print()
    
    return results

def analyze_residuals(y_true, y_pred, title):
    """Analyze model residuals for patterns"""
    residuals = y_true - y_pred
    
    print(f"=== RESIDUAL ANALYSIS: {title} ===")
    print(f"Mean residual: ${np.mean(residuals):.2f}")
    print(f"Std residual: ${np.std(residuals):.2f}")
    print(f"Max absolute residual: ${np.max(np.abs(residuals)):.2f}")
    
    # Look for systematic biases
    print(f"Residuals > $50: {np.sum(np.abs(residuals) > 50)} cases")
    print(f"Residuals > $100: {np.sum(np.abs(residuals) > 100)} cases")
    print()

def identify_outliers(df, results):
    """Identify cases where all models perform poorly"""
    
    print("=== OUTLIER ANALYSIS ===\n")
    
    # Calculate average absolute error across all models
    all_errors = []
    for model_name, model_results in results.items():
        if 'predictions' in model_results:
            errors = np.abs(df['reimbursement'].values - model_results['predictions'])
            all_errors.append(errors)
    
    if all_errors:
        avg_errors = np.mean(all_errors, axis=0)
        
        # Find the worst cases
        worst_indices = np.argsort(avg_errors)[-10:]
        
        print("Top 10 Hardest Cases to Predict:")
        for i, idx in enumerate(worst_indices[::-1]):
            row = df.iloc[idx]
            error = avg_errors[idx]
            print(f"{i+1:2d}. Days: {int(row['trip_duration_days']):2d}, "
                  f"Miles: {row['miles_traveled']:4.0f}, "
                  f"Receipts: ${row['total_receipts_amount']:7.2f}, "
                  f"Actual: ${row['reimbursement']:7.2f}, "
                  f"Avg Error: ${error:.2f}")
        print()

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Training cases: {len(train_df)}")
    print(f"Test cases: {len(test_df)}")
    print()
    
    # Basic statistics
    print("=== BASIC DATA STATISTICS ===\n")
    print("Training Data Summary:")
    print(train_df.describe())
    print()
    
    # Engineer features
    print("Engineering features...")
    train_features = engineer_features(train_df)
    test_features = engineer_features(test_df)
    
    # Feature columns (exclude original target)
    feature_cols = [col for col in train_features.columns if col != 'reimbursement']
    
    X_train = train_features[feature_cols]
    y_train = train_features['reimbursement']
    X_test = test_features[feature_cols]
    y_test = test_features['reimbursement']
    
    print(f"Created {len(feature_cols)} features")
    print()
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(X_train, y_train, feature_cols)
    
    # Test models
    results = test_models(X_train, y_train, X_test, y_test)
    
    # Analyze residuals for best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_model_results = results[best_model_name]
    
    print(f"Best model: {best_model_name}")
    analyze_residuals(y_test, best_model_results['predictions'], best_model_name)
    
    # Identify outliers
    identify_outliers(test_features, results)
    
    # Save results
    print("Saving analysis results...")
    
    # Save feature importance
    importance_df.to_csv('feature_importance.csv', index=False)
    
    # Save predictions for further analysis
    predictions_df = test_features[['trip_duration_days', 'miles_traveled', 
                                   'total_receipts_amount', 'reimbursement']].copy()
    
    for model_name, model_results in results.items():
        if 'predictions' in model_results:
            predictions_df[f'{model_name}_prediction'] = model_results['predictions']
            predictions_df[f'{model_name}_error'] = (predictions_df['reimbursement'] - 
                                                    model_results['predictions'])
    
    predictions_df.to_csv('model_predictions.csv', index=False)
    
    print("Analysis complete! Check feature_importance.csv and model_predictions.csv for detailed results.")

if __name__ == "__main__":
    main() 