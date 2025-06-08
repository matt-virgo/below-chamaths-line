#!/usr/bin/env python3
"""
Gradient Boost Model for Expense Reimbursement Prediction
Based on employee interview insights from INTERVIEWS.md

Features designed to capture:
1. Human-observed patterns (efficiency bonuses, sweet spots, penalties)
2. Programmatic patterns (algorithmic calculations, thresholds, interaction effects)
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import math

def create_features(data):
    """
    Create 20 features based on interview insights
    10 human-observed patterns + 10 programmatic/algorithmic patterns
    """
    # Extract base features from JSON structure
    trip_duration_days = np.array([d['input']['trip_duration_days'] for d in data])
    miles_traveled = np.array([d['input']['miles_traveled'] for d in data])
    total_receipts_amount = np.array([d['input']['total_receipts_amount'] for d in data])
    
    features = {}
    
    # ===== HUMAN-OBSERVED PATTERNS (Features 1-10) =====
    
    # 1. Miles per day efficiency (Marcus, Kevin mentioned 180-220 sweet spot)
    features['miles_per_day'] = miles_traveled / trip_duration_days
    
    # 2. Daily spending rate (Marcus mentioned $60-90 per day patterns)
    features['receipts_per_day'] = total_receipts_amount / trip_duration_days
    
    # 3. Sweet spot trip duration (5-6 days mentioned multiple times)
    features['is_sweet_spot_duration'] = ((trip_duration_days >= 4) & (trip_duration_days <= 6)).astype(int)
    
    # 4. Efficiency bonus indicator (Kevin's 180-220 miles per day sweet spot)
    features['efficiency_bonus'] = ((features['miles_per_day'] >= 180) & (features['miles_per_day'] <= 220)).astype(int)
    
    # 5. High mileage long trip indicator (mentioned by multiple interviewees)
    features['high_mileage_long_trip'] = ((miles_traveled > 600) & (trip_duration_days >= 7)).astype(int)
    
    # 6. Low receipt penalty (Dave mentioned small receipts get penalized)
    features['low_receipt_penalty'] = (total_receipts_amount < 50).astype(int)
    
    # 7. Vacation penalty (Kevin's 8+ day high spending penalty)
    features['vacation_penalty'] = ((trip_duration_days >= 8) & (features['receipts_per_day'] > 150)).astype(int)
    
    # 8. Modest spending bonus (Marcus mentioned $60-90 per day works well)
    features['modest_spending'] = ((features['receipts_per_day'] >= 60) & (features['receipts_per_day'] <= 120)).astype(int)
    
    # 9. High efficiency short trip (quick trips with high mileage)
    features['efficient_short_trip'] = ((trip_duration_days <= 3) & (features['miles_per_day'] > 150)).astype(int)
    
    # 10. Balanced trip indicator (Kevin's "sweet spot combo")
    features['balanced_trip'] = ((trip_duration_days == 5) & 
                          (features['miles_per_day'] >= 180) & 
                          (features['receipts_per_day'] <= 100)).astype(int)
    
    # ===== PROGRAMMATIC/ALGORITHMIC PATTERNS (Features 11-20) =====
    
    # 11. Logarithmic mileage scaling (Lisa mentioned non-linear mileage curves)
    features['log_miles'] = np.log1p(miles_traveled)
    
    # 12. Quadratic duration effect (captures non-linear trip length effects)
    features['duration_squared'] = trip_duration_days ** 2
    
    # 13. Receipt amount logarithm (handles diminishing returns on spending)
    features['log_receipts'] = np.log1p(total_receipts_amount)
    
    # 14. Interaction: miles × duration (algorithmic combination effect)
    features['miles_duration_interaction'] = miles_traveled * trip_duration_days
    
    # 15. Interaction: receipts × duration (spending efficiency over time)
    features['receipts_duration_interaction'] = total_receipts_amount * trip_duration_days
    
    # 16. Hash-based feature (pseudo-randomization that engineers might implement)
    features['hash_feature'] = (miles_traveled.astype(int) * 31 + 
                         trip_duration_days * 17 + 
                         total_receipts_amount.astype(int) * 13) % 100
    
    # 17. Threshold step function (mimics programmatic decision trees)
    features['threshold_step'] = np.where(miles_traveled > 300, 
                                   np.where(miles_traveled > 600, 2, 1), 0)
    
    # 18. Modular arithmetic feature (engineers love modular patterns)
    features['modular_pattern'] = (trip_duration_days % 3) + (miles_traveled.astype(int) % 7) * 0.1
    
    # 19. Complex ratio (sophisticated algorithmic calculation)
    features['complex_ratio'] = (miles_traveled * np.sqrt(trip_duration_days)) / (total_receipts_amount + 1)
    
    # 20. Binned efficiency score (programmatic efficiency categorization)
    efficiency_score = miles_traveled / (trip_duration_days * (total_receipts_amount + 100))
    # Handle the binning more carefully to avoid errors
    try:
        features['binned_efficiency'] = pd.qcut(efficiency_score, q=5, labels=False, duplicates='drop')
    except:
        # If qcut fails, use simple binning
        features['binned_efficiency'] = np.digitize(efficiency_score, np.percentile(efficiency_score, [20, 40, 60, 80])) - 1
    
    # Convert to numpy array
    feature_names = list(features.keys())
    feature_matrix = np.column_stack([features[name] for name in feature_names])
    
    return feature_matrix, feature_names

def load_data(filename):
    """Load and return data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_targets(data):
    """Extract target values from data"""
    return np.array([d['expected_output'] for d in data])

def main():
    print("=== Gradient Boost Model for Expense Reimbursement ===")
    print("Based on employee interview insights\n")
    
    # Load data
    print("Loading data...")
    train_data = load_data('train_cases.json')
    test_data = load_data('test_cases.json')
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create features
    print("\nCreating features...")
    X_train, feature_names = create_features(train_data)
    X_test, _ = create_features(test_data)
    y_train = extract_targets(train_data)
    y_test = extract_targets(test_data)
    
    print(f"Features created: {len(feature_names)}")
    print("Feature list:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    # Train model
    print("\nTraining Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    # Cross-validation on training data
    print("\nPerforming 5-fold cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(gb_model, X_train, y_train, cv=kfold, 
                               scoring='neg_mean_squared_error', n_jobs=-1)
    
    cv_rmse = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})")
    
    # Train final model
    gb_model.fit(X_train, y_train)
    
    # Predictions
    train_pred = gb_model.predict(X_train)
    test_pred = gb_model.predict(X_test)
    
    # Training metrics
    print(f"\n=== TRAINING SET RESULTS ===")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_train, train_pred)):.2f}")
    print(f"MAE:  {mean_absolute_error(y_train, train_pred):.2f}")
    print(f"R²:   {r2_score(y_train, train_pred):.4f}")
    
    # Test metrics (holdout evaluation)
    print(f"\n=== HOLDOUT TEST SET RESULTS ===")
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"RMSE: {test_rmse:.2f}")
    print(f"MAE:  {test_mae:.2f}")
    print(f"R²:   {test_r2:.4f}")
    
    # Feature importance analysis
    print(f"\n=== FEATURE IMPORTANCE ===")
    importances = gb_model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 most important features:")
    for i, (name, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i:2d}. {name:<25} {importance:.4f}")
    
    # Save model and feature importance
    print(f"\nSaving model and results...")
    joblib.dump(gb_model, 'xgboost_interview_model.pkl')
    
    # Save detailed feature importance
    importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    importance_df.to_csv('feature_importance_interview.csv', index=False)
    
    # Save predictions for analysis
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': test_pred,
        'error': y_test - test_pred,
        'abs_error': np.abs(y_test - test_pred)
    })
    predictions_df.to_csv('xgboost_interview_predictions.csv', index=False)
    
    # Performance summary
    print(f"\n=== SUMMARY ===")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f}")
    print(f"Holdout test RMSE:     {test_rmse:.2f}")
    print(f"Holdout test R²:       {test_r2:.4f}")
    print(f"Mean absolute error:   {test_mae:.2f}")
    
    # Sample predictions for verification
    print(f"\n=== SAMPLE PREDICTIONS ===")
    print("Actual vs Predicted (first 10 test cases):")
    for i in range(min(10, len(y_test))):
        print(f"  {y_test[i]:8.2f} vs {test_pred[i]:8.2f} (error: {y_test[i] - test_pred[i]:+7.2f})")
    
    return {
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = main() 