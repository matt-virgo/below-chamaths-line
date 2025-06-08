#!/usr/bin/env python3
"""
Mega Feature Gradient Boost Model for Expense Reimbursement Prediction
1000 features total:
- 500 based on employee interview insights
- 500 based on software engineering/algorithmic patterns
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import math

def create_interview_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Create 500 features based on employee interview insights
    Expanding on patterns from Marcus, Lisa, Dave, Jennifer, and Kevin
    """
    features = {}
    
    # Base derived metrics
    miles_per_day = miles_traveled / trip_duration_days
    receipts_per_day = total_receipts_amount / trip_duration_days
    
    # === EFFICIENCY PATTERNS (Kevin's obsession) ===
    # Kevin mentioned 180-220 sweet spot, let's explore many efficiency ranges
    for i, (low, high) in enumerate([(150, 180), (160, 190), (170, 200), (180, 220), 
                                    (190, 230), (200, 250), (220, 280), (250, 300)]):
        features[f'efficiency_range_{i}'] = ((miles_per_day >= low) & (miles_per_day <= high)).astype(int)
    
    # Multiple efficiency bonus thresholds
    for threshold in [100, 120, 150, 180, 200, 220, 250, 300, 350, 400]:
        features[f'high_efficiency_{threshold}'] = (miles_per_day > threshold).astype(int)
        features[f'low_efficiency_{threshold}'] = (miles_per_day < threshold).astype(int)
    
    # === SPENDING PATTERNS (Marcus's theories) ===
    # Marcus mentioned $60-90 sweet spot, let's explore many spending ranges
    for i, (low, high) in enumerate([(40, 60), (50, 80), (60, 90), (70, 100), 
                                    (80, 120), (90, 130), (100, 150), (120, 180)]):
        features[f'spending_range_{i}'] = ((receipts_per_day >= low) & (receipts_per_day <= high)).astype(int)
    
    # Daily spending thresholds
    for threshold in [30, 50, 75, 100, 125, 150, 200, 250, 300]:
        features[f'high_spending_{threshold}'] = (receipts_per_day > threshold).astype(int)
        features[f'low_spending_{threshold}'] = (receipts_per_day < threshold).astype(int)
    
    # === TRIP DURATION PATTERNS (Multiple mentions) ===
    # Sweet spots and penalties for different durations
    for duration in range(1, 15):
        features[f'exact_duration_{duration}'] = (trip_duration_days == duration).astype(int)
    
    # Duration ranges (everyone mentioned different sweet spots)
    for i, (low, high) in enumerate([(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), 
                                    (6, 8), (7, 10), (8, 12), (10, 14)]):
        features[f'duration_range_{i}'] = ((trip_duration_days >= low) & (trip_duration_days <= high)).astype(int)
    
    # === MILEAGE PATTERNS (Lisa's non-linear observations) ===
    # Different mileage thresholds and ranges
    for threshold in [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200]:
        features[f'high_mileage_{threshold}'] = (miles_traveled > threshold).astype(int)
        features[f'low_mileage_{threshold}'] = (miles_traveled < threshold).astype(int)
    
    # Mileage ranges
    for i, (low, high) in enumerate([(0, 100), (50, 200), (100, 300), (200, 500), 
                                    (300, 600), (500, 800), (600, 1000), (800, 1200)]):
        features[f'mileage_range_{i}'] = ((miles_traveled >= low) & (miles_traveled <= high)).astype(int)
    
    # === COMBINATION PATTERNS (Kevin's sweet spot combos) ===
    # Various combinations of trip characteristics
    for duration in [3, 4, 5, 6, 7]:
        for miles_thresh in [150, 180, 200, 220]:
            for spend_thresh in [80, 100, 120]:
                features[f'combo_{duration}d_{miles_thresh}m_{spend_thresh}s'] = (
                    (trip_duration_days == duration) & 
                    (miles_per_day >= miles_thresh) & 
                    (receipts_per_day <= spend_thresh)
                ).astype(int)
    
    # === PENALTY PATTERNS (Dave and Kevin's observations) ===
    # Vacation penalties (long trips with high spending)
    for duration_thresh in [7, 8, 9, 10]:
        for spending_thresh in [120, 150, 180, 200]:
            features[f'vacation_penalty_{duration_thresh}d_{spending_thresh}s'] = (
                (trip_duration_days >= duration_thresh) & (receipts_per_day > spending_thresh)
            ).astype(int)
    
    # Low receipt penalties
    for threshold in [20, 30, 40, 50, 60, 75]:
        features[f'low_receipt_penalty_{threshold}'] = (total_receipts_amount < threshold).astype(int)
    
    # === EFFICIENCY RATIOS (Multiple variations) ===
    # Different ways to measure efficiency (Kevin would love these)
    features['miles_per_dollar'] = miles_traveled / (total_receipts_amount + 1)
    features['dollars_per_mile'] = total_receipts_amount / (miles_traveled + 1)
    features['efficiency_score_1'] = miles_traveled / (trip_duration_days * (total_receipts_amount + 100))
    features['efficiency_score_2'] = (miles_traveled * trip_duration_days) / (total_receipts_amount + 1)
    features['efficiency_score_3'] = miles_per_day / (receipts_per_day + 1)
    
    # === SEASONAL/TIMING PATTERNS (Kevin's lunar theories) ===
    # Pseudo-seasonal effects based on trip characteristics
    features['trip_type_hash'] = (trip_duration_days * 7 + miles_traveled.astype(int) % 30) % 12
    for season in range(12):
        features[f'pseudo_season_{season}'] = (features['trip_type_hash'] == season).astype(int)
    
    # === SPENDING EFFICIENCY PATTERNS ===
    # Marcus mentioned different spending efficiencies
    features['spending_efficiency_1'] = total_receipts_amount / trip_duration_days
    features['spending_efficiency_2'] = total_receipts_amount / (miles_traveled + 1)
    features['spending_efficiency_3'] = total_receipts_amount / ((trip_duration_days * miles_traveled) + 1)
    
    # === QUARTILE AND PERCENTILE PATTERNS ===
    # Binning strategies that engineers might use
    for metric_name, metric_values in [('miles', miles_traveled), ('duration', trip_duration_days), 
                                      ('receipts', total_receipts_amount), ('miles_per_day', miles_per_day)]:
        try:
            quartiles = pd.qcut(metric_values, q=4, labels=False, duplicates='drop')
            for q in range(4):
                features[f'{metric_name}_quartile_{q}'] = (quartiles == q).astype(int)
        except:
            # Fallback if qcut fails
            percentiles = np.percentile(metric_values, [25, 50, 75])
            features[f'{metric_name}_quartile_0'] = (metric_values <= percentiles[0]).astype(int)
            features[f'{metric_name}_quartile_1'] = ((metric_values > percentiles[0]) & (metric_values <= percentiles[1])).astype(int)
            features[f'{metric_name}_quartile_2'] = ((metric_values > percentiles[1]) & (metric_values <= percentiles[2])).astype(int)
            features[f'{metric_name}_quartile_3'] = (metric_values > percentiles[2]).astype(int)
    
    # === BUSINESS LOGIC PATTERNS ===
    # Patterns that would make business sense
    features['weekend_trip'] = (trip_duration_days % 7 == 0).astype(int)  # Weekly trips
    features['business_week'] = (trip_duration_days == 5).astype(int)
    features['long_weekend'] = (trip_duration_days == 3).astype(int)
    features['extended_trip'] = (trip_duration_days > 10).astype(int)
    
    # Daily averages and their patterns
    avg_daily_miles = miles_traveled / trip_duration_days
    avg_daily_spending = total_receipts_amount / trip_duration_days
    features['balanced_daily_ratio'] = (avg_daily_miles / (avg_daily_spending + 1))
    
    # === MATHEMATICAL COMBINATIONS FROM INTERVIEWS ===
    # Kevin would definitely try these mathematical combinations
    features['miles_duration_product'] = miles_traveled * trip_duration_days
    features['receipts_duration_product'] = total_receipts_amount * trip_duration_days
    features['miles_receipts_ratio'] = miles_traveled / (total_receipts_amount + 1)
    features['duration_efficiency'] = trip_duration_days / (miles_per_day + 1)
    
    # Pad to exactly 500 features by adding variations and interactions
    current_count = len(features)
    remaining = 500 - current_count
    
    # Add more complex combinations to reach 500
    base_metrics = [miles_traveled, trip_duration_days, total_receipts_amount, miles_per_day, receipts_per_day]
    metric_names = ['miles', 'duration', 'receipts', 'miles_per_day', 'receipts_per_day']
    
    for i in range(remaining):
        if i < len(base_metrics):
            # Square and cube transformations
            features[f'interview_transform_{i}_sq'] = base_metrics[i] ** 2
            features[f'interview_transform_{i}_cube'] = base_metrics[i] ** 3
        else:
            # Additional interaction terms
            idx1 = i % len(base_metrics)
            idx2 = (i + 1) % len(base_metrics)
            features[f'interview_interaction_{i}'] = base_metrics[idx1] * base_metrics[idx2]
    
    # Trim to exactly 500 if we went over
    feature_names = list(features.keys())[:500]
    return {name: features[name] for name in feature_names}

def create_engineering_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Create 500 features based on software engineering/algorithmic patterns
    """
    features = {}
    
    # === MATHEMATICAL TRANSFORMATIONS ===
    base_values = [trip_duration_days, miles_traveled, total_receipts_amount]
    base_names = ['duration', 'miles', 'receipts']
    
    # Logarithmic transformations (various bases)
    for i, (values, name) in enumerate(zip(base_values, base_names)):
        features[f'log_{name}'] = np.log1p(values)
        features[f'log2_{name}'] = np.log2(values + 1)
        features[f'log10_{name}'] = np.log10(values + 1)
        features[f'sqrt_{name}'] = np.sqrt(values)
        features[f'cbrt_{name}'] = np.cbrt(values)
    
    # Power transformations
    for power in [0.25, 0.5, 1.5, 2, 2.5, 3, 4]:
        for i, (values, name) in enumerate(zip(base_values, base_names)):
            features[f'{name}_power_{power}'] = np.power(values, power)
    
    # === TRIGONOMETRIC FEATURES ===
    # Engineers love periodic patterns
    for values, name in zip(base_values, base_names):
        normalized = values / (np.max(values) + 1)  # Normalize to [0,1]
        features[f'sin_{name}'] = np.sin(2 * np.pi * normalized)
        features[f'cos_{name}'] = np.cos(2 * np.pi * normalized)
        features[f'tan_{name}'] = np.tan(np.pi * normalized)
        
        # Multiple frequencies
        for freq in [2, 3, 4, 5]:
            features[f'sin_{name}_freq_{freq}'] = np.sin(2 * np.pi * freq * normalized)
            features[f'cos_{name}_freq_{freq}'] = np.cos(2 * np.pi * freq * normalized)
    
    # === HASH FUNCTIONS ===
    # Various hash-based features (engineers love deterministic randomness)
    for prime1, prime2, prime3 in [(31, 17, 13), (37, 23, 19), (41, 29, 11), 
                                   (43, 31, 7), (47, 37, 3), (53, 41, 2)]:
        hash_val = (miles_traveled.astype(int) * prime1 + 
                   trip_duration_days * prime2 + 
                   total_receipts_amount.astype(int) * prime3)
        
        features[f'hash_{prime1}_{prime2}_{prime3}'] = hash_val % 100
        features[f'hash_mod_{prime1}_{prime2}_{prime3}'] = hash_val % 17
        features[f'hash_sin_{prime1}'] = np.sin(hash_val / 100.0)
        features[f'hash_cos_{prime1}'] = np.cos(hash_val / 100.0)
    
    # === MODULAR ARITHMETIC ===
    # Different moduli for different patterns
    for mod_val in [3, 5, 7, 11, 13, 17, 19, 23]:
        features[f'duration_mod_{mod_val}'] = trip_duration_days % mod_val
        features[f'miles_mod_{mod_val}'] = miles_traveled.astype(int) % mod_val
        features[f'receipts_mod_{mod_val}'] = total_receipts_amount.astype(int) % mod_val
    
    # === POLYNOMIAL FEATURES ===
    # Various polynomial combinations
    for i in range(5):
        for j in range(5):
            for k in range(5):
                if i + j + k <= 4 and i + j + k > 0:  # Up to degree 4
                    features[f'poly_{i}_{j}_{k}'] = (trip_duration_days ** i) * (miles_traveled ** j) * (total_receipts_amount ** k)
    
    # === THRESHOLD FUNCTIONS ===
    # Step functions at various thresholds
    duration_thresholds = [1, 2, 3, 5, 7, 10, 14]
    miles_thresholds = [50, 100, 200, 300, 500, 800, 1000]
    receipt_thresholds = [20, 50, 100, 200, 500, 1000, 2000]
    
    for i, thresh in enumerate(duration_thresholds):
        features[f'duration_step_{i}'] = (trip_duration_days > thresh).astype(int)
        
    for i, thresh in enumerate(miles_thresholds):
        features[f'miles_step_{i}'] = (miles_traveled > thresh).astype(int)
        
    for i, thresh in enumerate(receipt_thresholds):
        features[f'receipts_step_{i}'] = (total_receipts_amount > thresh).astype(int)
    
    # === INTERACTION TERMS ===
    # All pairwise and some 3-way interactions
    interactions = [
        ('duration_miles', trip_duration_days * miles_traveled),
        ('duration_receipts', trip_duration_days * total_receipts_amount),
        ('miles_receipts', miles_traveled * total_receipts_amount),
        ('duration_miles_receipts', trip_duration_days * miles_traveled * total_receipts_amount),
        ('duration_sq_miles', (trip_duration_days ** 2) * miles_traveled),
        ('miles_sq_receipts', (miles_traveled ** 2) * total_receipts_amount),
        ('duration_miles_sq', trip_duration_days * (miles_traveled ** 2)),
    ]
    
    for name, values in interactions:
        features[f'interaction_{name}'] = values
    
    # === STATISTICAL AGGREGATIONS ===
    # Rolling statistics and percentile-based features
    all_values = np.column_stack([trip_duration_days, miles_traveled, total_receipts_amount])
    
    features['row_mean'] = np.mean(all_values, axis=1)
    features['row_std'] = np.std(all_values, axis=1)
    features['row_min'] = np.min(all_values, axis=1)
    features['row_max'] = np.max(all_values, axis=1)
    features['row_range'] = features['row_max'] - features['row_min']
    features['row_median'] = np.median(all_values, axis=1)
    
    # === COMPLEX MATHEMATICAL FUNCTIONS ===
    # More sophisticated mathematical transformations
    features['harmonic_mean'] = 3.0 / (1.0/(trip_duration_days + 1) + 1.0/(miles_traveled + 1) + 1.0/(total_receipts_amount + 1))
    features['geometric_mean'] = np.power(trip_duration_days * miles_traveled * total_receipts_amount, 1/3)
    
    # Exponential and hyperbolic functions
    for values, name in zip(base_values, base_names):
        normalized = values / (np.max(values) + 1)
        features[f'exp_{name}'] = np.exp(-normalized)  # Negative exp to avoid overflow
        features[f'sinh_{name}'] = np.sinh(normalized)
        features[f'cosh_{name}'] = np.cosh(normalized)
        features[f'tanh_{name}'] = np.tanh(normalized)
    
    # === BINNING AND DISCRETIZATION ===
    # Various binning strategies
    for n_bins in [3, 5, 8, 10]:
        for values, name in zip(base_values, base_names):
            try:
                bins = pd.cut(values, bins=n_bins, labels=False)
                for bin_idx in range(n_bins):
                    features[f'{name}_bin_{n_bins}_{bin_idx}'] = (bins == bin_idx).astype(int)
            except:
                # Fallback binning
                percentiles = np.linspace(0, 100, n_bins + 1)
                thresholds = np.percentile(values, percentiles)
                bins = np.digitize(values, thresholds[1:-1])
                for bin_idx in range(n_bins):
                    features[f'{name}_bin_{n_bins}_{bin_idx}'] = (bins == bin_idx).astype(int)
    
    # === FOURIER-LIKE FEATURES ===
    # Frequency domain transformations
    for i, (values, name) in enumerate(zip(base_values, base_names)):
        for freq in range(1, 6):
            features[f'fourier_{name}_sin_{freq}'] = np.sin(2 * np.pi * freq * values / np.max(values))
            features[f'fourier_{name}_cos_{freq}'] = np.cos(2 * np.pi * freq * values / np.max(values))
    
    # === ALGORITHMIC PATTERNS ===
    # Patterns that would emerge from algorithmic thinking
    features['bit_pattern_duration'] = trip_duration_days & 7  # Last 3 bits
    features['bit_pattern_miles'] = miles_traveled.astype(int) & 15  # Last 4 bits
    features['bit_pattern_receipts'] = total_receipts_amount.astype(int) & 31  # Last 5 bits
    
    # XOR patterns
    features['xor_duration_miles'] = trip_duration_days ^ miles_traveled.astype(int)
    features['xor_all'] = trip_duration_days ^ miles_traveled.astype(int) ^ total_receipts_amount.astype(int)
    
    # Fill remaining slots to reach exactly 500 features
    current_count = len(features)
    remaining = 500 - current_count
    
    # Add more complex derived features to fill remaining slots
    for i in range(remaining):
        # Generate additional polynomial and interaction features
        deg1 = i % 3 + 1
        deg2 = (i // 3) % 3 + 1
        idx1 = i % 3
        idx2 = (i + 1) % 3
        
        val1 = base_values[idx1] ** deg1
        val2 = base_values[idx2] ** deg2
        features[f'eng_complex_{i}'] = val1 * val2 / (val1 + val2 + 1)
    
    # Trim to exactly 500 if needed
    feature_names = list(features.keys())[:500]
    return {name: features[name] for name in feature_names}

def create_mega_features(data):
    """
    Create all 1000 features: 500 interview + 500 engineering
    """
    # Extract base features
    trip_duration_days = np.array([d['input']['trip_duration_days'] for d in data])
    miles_traveled = np.array([d['input']['miles_traveled'] for d in data])
    total_receipts_amount = np.array([d['input']['total_receipts_amount'] for d in data])
    
    print("Creating interview-based features (500)...")
    interview_features = create_interview_features(trip_duration_days, miles_traveled, total_receipts_amount)
    
    print("Creating engineering-based features (500)...")
    engineering_features = create_engineering_features(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Combine all features
    all_features = {**interview_features, **engineering_features}
    
    # Convert to numpy array
    feature_names = list(all_features.keys())
    feature_matrix = np.column_stack([all_features[name] for name in feature_names])
    
    print(f"Total features created: {len(feature_names)}")
    print(f"Interview features: {len(interview_features)}")
    print(f"Engineering features: {len(engineering_features)}")
    
    return feature_matrix, feature_names

def load_data(filename):
    """Load and return data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_targets(data):
    """Extract target values from data"""
    return np.array([d['expected_output'] for d in data])

def main():
    print("=== MEGA FEATURE Gradient Boost Model ===")
    print("1000 features: 500 interview + 500 engineering\n")
    
    # Load data
    print("Loading data...")
    train_data = load_data('train_cases.json')
    test_data = load_data('test_cases.json')
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create mega features
    print("\nCreating 1000 features...")
    X_train, feature_names = create_mega_features(train_data)
    X_test, _ = create_mega_features(test_data)
    y_train = extract_targets(train_data)
    y_test = extract_targets(test_data)
    
    print(f"\nFeature matrix shape: {X_train.shape}")
    
    # Train model with regularization for high-dimensional data
    print("\nTraining Gradient Boosting model...")
    gb_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,  # Lower learning rate for stability
        max_depth=4,         # Shallower trees to prevent overfitting
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.7,       # More aggressive subsampling
        max_features=0.3,    # Only use 30% of features per tree
        random_state=42
    )
    
    # Cross-validation
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
    
    # Test metrics
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
    
    print("Top 20 most important features:")
    for i, (name, importance) in enumerate(feature_importance[:20], 1):
        feature_type = "Interview" if any(keyword in name.lower() for keyword in 
                                        ['efficiency', 'spending', 'duration', 'combo', 'penalty', 'interview']) else "Engineering"
        print(f"  {i:2d}. {name:<35} {importance:.4f} ({feature_type})")
    
    # Analyze feature type distribution in top features
    top_50_features = feature_importance[:50]
    interview_count = sum(1 for name, _ in top_50_features 
                         if any(keyword in name.lower() for keyword in 
                               ['efficiency', 'spending', 'duration', 'combo', 'penalty', 'interview']))
    engineering_count = 50 - interview_count
    
    print(f"\nTop 50 features breakdown:")
    print(f"Interview-based: {interview_count} ({interview_count/50*100:.1f}%)")
    print(f"Engineering-based: {engineering_count} ({engineering_count/50*100:.1f}%)")
    
    # Save results
    print(f"\nSaving model and results...")
    joblib.dump(gb_model, 'mega_feature_model.pkl')
    
    # Save feature importance
    importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    importance_df['feature_type'] = importance_df['feature'].apply(
        lambda x: 'Interview' if any(keyword in x.lower() for keyword in 
                                   ['efficiency', 'spending', 'duration', 'combo', 'penalty', 'interview'])
                  else 'Engineering'
    )
    importance_df.to_csv('mega_feature_importance.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': test_pred,
        'error': y_test - test_pred,
        'abs_error': np.abs(y_test - test_pred)
    })
    predictions_df.to_csv('mega_feature_predictions.csv', index=False)
    
    # Performance summary
    print(f"\n=== SUMMARY ===")
    print(f"Features used: {len(feature_names)}")
    print(f"Cross-validation RMSE: {cv_rmse.mean():.2f}")
    print(f"Holdout test RMSE:     {test_rmse:.2f}")
    print(f"Holdout test R²:       {test_r2:.4f}")
    print(f"Mean absolute error:   {test_mae:.2f}")
    
    # Sample predictions
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
        'feature_importance': feature_importance,
        'total_features': len(feature_names)
    }

if __name__ == "__main__":
    results = main() 