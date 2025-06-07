#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
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

def create_ultra_features(df):
    """Create an exhaustive feature set for maximum precision"""
    features_df = df.copy()
    
    # Original features
    trip_days = features_df['trip_duration_days']
    miles = features_df['miles_traveled']
    receipts = features_df['total_receipts_amount']
    
    # Basic derived
    features_df['miles_per_day'] = miles / trip_days
    features_df['receipts_per_day'] = receipts / trip_days
    
    # Most important features from previous analysis
    features_df['total_trip_value'] = trip_days * miles * receipts
    features_df['receipts_log'] = np.log1p(receipts)
    features_df['receipts_sqrt'] = np.sqrt(receipts)
    features_df['receipts_squared'] = receipts ** 2
    
    # Lucky cents (validated as important)
    features_df['receipts_cents'] = (receipts * 100) % 100
    features_df['has_lucky_cents'] = ((features_df['receipts_cents'] == 49) | 
                                     (features_df['receipts_cents'] == 99)).astype(int)
    
    # Interactions
    features_df['miles_receipts'] = miles * receipts
    features_df['days_receipts'] = trip_days * receipts
    features_df['days_miles'] = trip_days * miles
    
    # Advanced transformations
    features_df['miles_log'] = np.log1p(miles)
    features_df['miles_sqrt'] = np.sqrt(miles)
    features_df['miles_squared'] = miles ** 2
    features_df['days_squared'] = trip_days ** 2
    
    # Ratios
    features_df['receipts_miles_ratio'] = receipts / (miles + 1)
    features_df['miles_days_ratio'] = miles / trip_days
    
    # Higher order interactions
    features_df['miles_per_day_squared'] = features_df['miles_per_day'] ** 2
    features_df['receipts_per_day_squared'] = features_df['receipts_per_day'] ** 2
    features_df['miles_receipts_per_day'] = features_df['miles_per_day'] * features_df['receipts_per_day']
    
    # Exponential features (normalized)
    features_df['receipts_exp_norm'] = np.exp(receipts / 2000) - 1
    features_df['miles_exp_norm'] = np.exp(miles / 1000) - 1
    
    # Trigonometric features
    features_df['receipts_sin'] = np.sin(receipts / 1000)
    features_df['miles_sin'] = np.sin(miles / 500)
    features_df['receipts_cos'] = np.cos(receipts / 1000)
    features_df['miles_cos'] = np.cos(miles / 500)
    
    # Polynomial combinations
    features_df['days_miles_receipts'] = trip_days * miles * receipts
    features_df['sqrt_days_miles_receipts'] = np.sqrt(trip_days * miles * receipts)
    features_df['log_days_miles_receipts'] = np.log1p(trip_days * miles * receipts)
    
    # Binned features
    features_df['receipts_bin'] = pd.cut(receipts, bins=20, labels=False)
    features_df['miles_bin'] = pd.cut(miles, bins=20, labels=False)
    features_df['days_bin'] = pd.cut(trip_days, bins=10, labels=False)
    
    # Remove target if present
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    return features_df[feature_cols]

class PrecisionEnsemble(BaseEstimator, RegressorMixin):
    """Custom ensemble designed for maximum precision"""
    
    def __init__(self):
        # Multiple models with different strengths
        self.models = {
            'xgb_1': xgb.XGBRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'xgb_2': xgb.XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=123
            ),
            'lgb_1': lgb.LGBMRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lgb_2': lgb.LGBMRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=123
            ),
            'gb_1': GradientBoostingRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.8,
                random_state=42
            ),
            'gb_2': GradientBoostingRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.9,
                random_state=123
            ),
            'rf_1': RandomForestRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'rf_2': RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=123
            ),
            'et': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.fitted_models = {}
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train each model
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_scaled, y)
            self.fitted_models[name] = model
            
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.fitted_models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average (give more weight to historically better models)
        weights = np.array([0.15, 0.15, 0.15, 0.15, 0.12, 0.12, 0.08, 0.05, 0.03])
        
        final_prediction = np.average(predictions, axis=0, weights=weights)
        return final_prediction

class StackedEnsemble(BaseEstimator, RegressorMixin):
    """Stacked ensemble with meta-learner"""
    
    def __init__(self):
        # Base models
        self.base_models = {
            'xgb': xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42),
            'rf': RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=300, max_depth=12, random_state=42)
        }
        
        # Meta-learner
        self.meta_model = GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=3, 
            learning_rate=0.1, 
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.fitted_base_models = {}
        
    def fit(self, X, y):
        from sklearn.model_selection import KFold
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate meta-features using cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training base model {name}...")
            
            # Cross-validation predictions for meta-features
            cv_preds = np.zeros(len(X))
            for train_idx, val_idx in kf.split(X_scaled):
                X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
                y_train_fold = y[train_idx]
                
                model_clone = type(model)(**model.get_params())
                model_clone.fit(X_train_fold, y_train_fold)
                cv_preds[val_idx] = model_clone.predict(X_val_fold)
            
            meta_features[:, i] = cv_preds
            
            # Train on full data for final model
            model.fit(X_scaled, y)
            self.fitted_base_models[name] = model
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        
        # Get base model predictions
        meta_features = np.zeros((len(X), len(self.fitted_base_models)))
        for i, (name, model) in enumerate(self.fitted_base_models.items()):
            meta_features[:, i] = model.predict(X_scaled)
        
        # Meta-model prediction
        return self.meta_model.predict(meta_features)

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Creating ultra-precision features...")
    X_train = create_ultra_features(train_df)
    X_test = create_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} features")
    
    # Test multiple ensemble approaches
    ensembles = {
        'Precision Ensemble': PrecisionEnsemble(),
        'Stacked Ensemble': StackedEnsemble()
    }
    
    results = {}
    
    for name, ensemble in ensembles.items():
        print(f"\n=== Training {name} ===")
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        
        # Predictions
        train_pred = ensemble.predict(X_train)
        test_pred = ensemble.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Precision metrics
        exact_matches = np.sum(np.abs(y_test - test_pred) < 0.01)
        close_matches_1 = np.sum(np.abs(y_test - test_pred) < 1.0)
        close_matches_5 = np.sum(np.abs(y_test - test_pred) < 5.0)
        close_matches_10 = np.sum(np.abs(y_test - test_pred) < 10.0)
        
        results[name] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': test_pred,
            'exact_matches': exact_matches,
            'close_matches_1': close_matches_1,
            'close_matches_5': close_matches_5,
            'close_matches_10': close_matches_10
        }
        
        print(f"  Train MAE: ${train_mae:.2f}")
        print(f"  Test MAE:  ${test_mae:.2f}")
        print(f"  Train R²:  {train_r2:.6f}")
        print(f"  Test R²:   {test_r2:.6f}")
        print(f"  Exact matches (±$0.01): {exact_matches}/{len(y_test)} ({exact_matches/len(y_test)*100:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches_1}/{len(y_test)} ({close_matches_1/len(y_test)*100:.1f}%)")
        print(f"  Close matches (±$5.00): {close_matches_5}/{len(y_test)} ({close_matches_5/len(y_test)*100:.1f}%)")
        print(f"  Close matches (±$10.00): {close_matches_10}/{len(y_test)} ({close_matches_10/len(y_test)*100:.1f}%)")
    
    # Find best model
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
    best_results = results[best_model_name]
    
    print(f"\n=== Best Model: {best_model_name} ===")
    print(f"Test MAE: ${best_results['test_mae']:.2f}")
    print(f"Test R²: {best_results['test_r2']:.6f}")
    print(f"Exact matches: {best_results['exact_matches']}/{len(y_test)} ({best_results['exact_matches']/len(y_test)*100:.1f}%)")
    
    # Save detailed results
    detailed_results = pd.DataFrame({
        'trip_duration_days': test_df['trip_duration_days'],
        'miles_traveled': test_df['miles_traveled'],
        'total_receipts_amount': test_df['total_receipts_amount'],
        'actual_reimbursement': y_test
    })
    
    for name, model_results in results.items():
        detailed_results[f'{name}_prediction'] = model_results['predictions']
        detailed_results[f'{name}_error'] = y_test - model_results['predictions']
        detailed_results[f'{name}_abs_error'] = np.abs(y_test - model_results['predictions'])
    
    detailed_results.to_csv('precision_ensemble_results.csv', index=False)
    
    # Analyze the hardest cases
    best_pred = best_results['predictions']
    errors = np.abs(y_test - best_pred)
    worst_indices = np.argsort(errors)[-10:]
    
    print(f"\nTop 10 Hardest Cases for {best_model_name}:")
    for i, idx in enumerate(worst_indices[::-1]):
        print(f"{i+1:2d}. Days: {test_df.iloc[idx]['trip_duration_days']:2.0f}, "
              f"Miles: {test_df.iloc[idx]['miles_traveled']:4.0f}, "
              f"Receipts: ${test_df.iloc[idx]['total_receipts_amount']:7.2f}, "
              f"Actual: ${y_test[idx]:7.2f}, "
              f"Predicted: ${best_pred[idx]:7.2f}, "
              f"Error: ${errors[idx]:.2f}")
    
    print(f"\nDetailed results saved to precision_ensemble_results.csv")

if __name__ == "__main__":
    main() 