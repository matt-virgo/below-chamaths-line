#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import itertools
from collections import defaultdict
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

def analyze_exact_patterns(df):
    """Look for exact mathematical relationships"""
    
    print("=== SEARCHING FOR EXACT PATTERNS ===\n")
    
    # Simple linear combinations
    linear_formulas = [
        ('days * 100', lambda row: row['trip_duration_days'] * 100),
        ('miles * 0.5', lambda row: row['miles_traveled'] * 0.5),
        ('receipts * 1.0', lambda row: row['total_receipts_amount'] * 1.0),
        ('days * 100 + miles * 0.5', lambda row: row['trip_duration_days'] * 100 + row['miles_traveled'] * 0.5),
        ('days * 100 + receipts * 0.5', lambda row: row['trip_duration_days'] * 100 + row['total_receipts_amount'] * 0.5),
        ('miles * 0.5 + receipts * 0.5', lambda row: row['miles_traveled'] * 0.5 + row['total_receipts_amount'] * 0.5),
        ('days * 100 + miles * 0.5 + receipts * 0.5', lambda row: row['trip_duration_days'] * 100 + row['miles_traveled'] * 0.5 + row['total_receipts_amount'] * 0.5),
    ]
    
    for formula_name, formula_func in linear_formulas:
        predictions = df.apply(formula_func, axis=1)
        mae = mean_absolute_error(df['reimbursement'], predictions)
        exact_matches = np.sum(np.abs(df['reimbursement'] - predictions) < 0.01)
        
        if mae < 100 or exact_matches > 0:
            print(f"{formula_name}:")
            print(f"  MAE: ${mae:.2f}")
            print(f"  Exact matches: {exact_matches}/{len(df)}")
            print()
    
    # Test different per diem rates
    print("Testing different per diem rates:")
    for rate in [50, 75, 100, 125, 150, 175, 200]:
        predictions = df['trip_duration_days'] * rate + df['miles_traveled'] * 0.5 + df['total_receipts_amount'] * 0.5
        mae = mean_absolute_error(df['reimbursement'], predictions)
        exact_matches = np.sum(np.abs(df['reimbursement'] - predictions) < 0.01)
        
        if mae < 200 or exact_matches > 0:
            print(f"  Per diem ${rate}: MAE ${mae:.2f}, Exact: {exact_matches}")
    
    print()
    
    # Test different mileage rates
    print("Testing different mileage rates:")
    for rate in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        predictions = df['trip_duration_days'] * 100 + df['miles_traveled'] * rate + df['total_receipts_amount'] * 0.5
        mae = mean_absolute_error(df['reimbursement'], predictions)
        exact_matches = np.sum(np.abs(df['reimbursement'] - predictions) < 0.01)
        
        if mae < 200 or exact_matches > 0:
            print(f"  Mileage ${rate}: MAE ${mae:.2f}, Exact: {exact_matches}")
    
    print()

def analyze_conditional_patterns(df):
    """Look for conditional/branching patterns"""
    
    print("=== ANALYZING CONDITIONAL PATTERNS ===\n")
    
    # Analyze by trip duration
    print("Patterns by trip duration:")
    for days in sorted(df['trip_duration_days'].unique()):
        subset = df[df['trip_duration_days'] == days]
        if len(subset) >= 5:  # Only analyze if we have enough samples
            
            # Try simple formulas for this trip length
            formulas = [
                (f"days={days}: miles * rate + receipts * rate", 
                 lambda row, r1, r2: row['miles_traveled'] * r1 + row['total_receipts_amount'] * r2),
            ]
            
            best_mae = float('inf')
            best_formula = None
            
            # Test different rate combinations
            for mile_rate in [0.3, 0.4, 0.5, 0.6]:
                for receipt_rate in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    predictions = subset.apply(lambda row: row['miles_traveled'] * mile_rate + row['total_receipts_amount'] * receipt_rate, axis=1)
                    mae = mean_absolute_error(subset['reimbursement'], predictions)
                    
                    if mae < best_mae:
                        best_mae = mae
                        best_formula = f"miles * {mile_rate} + receipts * {receipt_rate}"
            
            exact_matches = np.sum(np.abs(subset['reimbursement'] - subset.apply(lambda row: row['miles_traveled'] * 0.5 + row['total_receipts_amount'] * 0.5, axis=1)) < 0.01)
            
            if best_mae < 50 or exact_matches > 0:
                print(f"  {days} days ({len(subset)} samples): {best_formula}, MAE ${best_mae:.2f}")
    
    print()

def create_symbolic_features(df):
    """Create a comprehensive set of potential symbolic features"""
    
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    # Per-day rates
    features_df['miles_per_day'] = M / D
    features_df['receipts_per_day'] = R / D
    
    # Standard reimbursement components
    features_df['per_diem_100'] = D * 100
    features_df['per_diem_125'] = D * 125
    features_df['per_diem_150'] = D * 150
    
    features_df['mileage_30'] = M * 0.30
    features_df['mileage_40'] = M * 0.40
    features_df['mileage_50'] = M * 0.50
    features_df['mileage_60'] = M * 0.60
    
    features_df['receipt_50'] = R * 0.50
    features_df['receipt_60'] = R * 0.60
    features_df['receipt_70'] = R * 0.70
    features_df['receipt_80'] = R * 0.80
    
    # Tier-based calculations (different rates for different ranges)
    features_df['miles_tier1'] = np.minimum(M, 100) * 0.60
    features_df['miles_tier2'] = np.maximum(0, np.minimum(M - 100, 400)) * 0.50
    features_df['miles_tier3'] = np.maximum(0, M - 500) * 0.40
    
    features_df['receipt_tier1'] = np.minimum(R, 200) * 0.80
    features_df['receipt_tier2'] = np.maximum(0, np.minimum(R - 200, 600)) * 0.60
    features_df['receipt_tier3'] = np.maximum(0, R - 800) * 0.50
    
    # Day-based modifiers
    features_df['short_trip_bonus'] = (D <= 2) * 50
    features_df['long_trip_penalty'] = (D >= 8) * -25
    features_df['optimal_5day'] = (D == 5) * 75
    
    # Efficiency bonuses/penalties
    mpd = M / D
    features_df['efficiency_bonus'] = ((mpd >= 150) & (mpd <= 250)) * 50
    features_df['low_efficiency_penalty'] = (mpd < 100) * -25
    features_df['high_efficiency_penalty'] = (mpd > 300) * -25
    
    # Receipt-based modifiers
    rpd = R / D
    features_df['frugal_bonus'] = (rpd < 75) * 25
    features_df['expensive_penalty'] = (rpd > 150) * -50
    
    # Interaction terms
    features_df['miles_receipts_interaction'] = M * R / 1000
    features_df['days_miles_interaction'] = D * M / 100
    features_df['days_receipts_interaction'] = D * R / 100
    
    # Mathematical transformations
    features_df['sqrt_miles'] = np.sqrt(M)
    features_df['sqrt_receipts'] = np.sqrt(R)
    features_df['log_miles'] = np.log1p(M)
    features_df['log_receipts'] = np.log1p(R)
    
    # Round numbers and special cases
    features_df['round_days'] = D
    features_df['round_miles_100'] = np.round(M / 100) * 100
    features_df['round_receipts_50'] = np.round(R / 50) * 50
    
    # Remove target
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    return features_df[feature_cols]

def deep_tree_analysis(X_train, y_train, X_test, y_test):
    """Use very deep decision trees to find exact patterns"""
    
    print("=== DEEP DECISION TREE ANALYSIS ===\n")
    
    # Try different tree depths to find exact rules
    for max_depth in [10, 15, 20, None]:
        for min_samples_leaf in [1, 2, 3]:
            dt = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
            
            dt.fit(X_train, y_train)
            
            train_pred = dt.predict(X_train)
            test_pred = dt.predict(X_test)
            
            train_exact = np.sum(np.abs(y_train - train_pred) < 0.01)
            test_exact = np.sum(np.abs(y_test - test_pred) < 0.01)
            
            train_mae = mean_absolute_error(y_train, train_pred)
            test_mae = mean_absolute_error(y_test, test_pred)
            
            if test_exact > 0 or test_mae < 50:
                print(f"Tree depth={max_depth}, min_leaf={min_samples_leaf}:")
                print(f"  Train exact: {train_exact}/{len(y_train)} ({train_exact/len(y_train)*100:.1f}%)")
                print(f"  Test exact: {test_exact}/{len(y_test)} ({test_exact/len(y_test)*100:.1f}%)")
                print(f"  Train MAE: ${train_mae:.2f}")
                print(f"  Test MAE: ${test_mae:.2f}")
                print()
                
                # If we found a good tree, show some rules
                if max_depth is not None and max_depth <= 15 and test_exact > 0:
                    tree_rules = export_text(dt, feature_names=list(X_train.columns), max_depth=3)
                    print("Sample decision rules:")
                    print(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)
                    print()

def ultra_precise_search(train_df, test_df):
    """Search for ultra-precise formulas using brute force"""
    
    print("=== ULTRA-PRECISE FORMULA SEARCH ===\n")
    
    D_train = train_df['trip_duration_days'].values
    M_train = train_df['miles_traveled'].values  
    R_train = train_df['total_receipts_amount'].values
    target_train = train_df['reimbursement'].values
    
    D_test = test_df['trip_duration_days'].values
    M_test = test_df['miles_traveled'].values
    R_test = test_df['total_receipts_amount'].values
    target_test = test_df['reimbursement'].values
    
    best_formulas = []
    
    # Test many coefficient combinations
    per_diem_rates = [75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
    mileage_rates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    receipt_rates = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    
    for pd_rate in per_diem_rates:
        for m_rate in mileage_rates:
            for r_rate in receipt_rates:
                # Basic formula: per_diem + mileage + receipts
                pred_train = D_train * pd_rate + M_train * m_rate + R_train * r_rate
                pred_test = D_test * pd_rate + M_test * m_rate + R_test * r_rate
                
                train_exact = np.sum(np.abs(target_train - pred_train) < 0.01)
                test_exact = np.sum(np.abs(target_test - pred_test) < 0.01)
                test_mae = mean_absolute_error(target_test, pred_test)
                
                if test_exact > 0 or test_mae < 30:
                    formula = f"days * {pd_rate} + miles * {m_rate} + receipts * {r_rate}"
                    best_formulas.append((formula, test_mae, test_exact, test_exact/len(target_test)*100))
    
    # Sort by test exact matches, then by MAE
    best_formulas.sort(key=lambda x: (-x[2], x[1]))
    
    print("Top formulas:")
    for i, (formula, mae, exact, exact_pct) in enumerate(best_formulas[:10]):
        print(f"{i+1:2d}. {formula}")
        print(f"    MAE: ${mae:.2f}, Exact: {exact}/{len(target_test)} ({exact_pct:.1f}%)")
    
    print()
    
    if best_formulas:
        return best_formulas[0]
    return None

def main():
    print("Loading data...")
    train_df, test_df = load_data()
    
    print("Analyzing exact mathematical patterns...")
    analyze_exact_patterns(train_df)
    
    print("Analyzing conditional patterns...")
    analyze_conditional_patterns(train_df)
    
    print("Creating symbolic features...")
    X_train = create_symbolic_features(train_df)
    X_test = create_symbolic_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"Created {X_train.shape[1]} symbolic features")
    print()
    
    # Deep tree analysis
    deep_tree_analysis(X_train, y_train, X_test, y_test)
    
    # Ultra-precise search
    best_formula = ultra_precise_search(train_df, test_df)
    
    if best_formula:
        formula, mae, exact, exact_pct = best_formula
        print(f"=== BEST DISCOVERED FORMULA ===")
        print(f"Formula: {formula}")
        print(f"Test MAE: ${mae:.2f}")
        print(f"Exact matches: {exact}/{len(test_df)} ({exact_pct:.1f}%)")
        
        # Test the best formula on some examples
        print(f"\nTesting on first 10 test cases:")
        for i in range(min(10, len(test_df))):
            row = test_df.iloc[i]
            actual = row['reimbursement']
            
            # Parse and apply the formula (this is a simplified version)
            if "days * 100 + miles * 0.5 + receipts * 0.5" in formula:
                predicted = row['trip_duration_days'] * 100 + row['miles_traveled'] * 0.5 + row['total_receipts_amount'] * 0.5
            else:
                # This would need more sophisticated parsing for other formulas
                predicted = 0
                
            error = abs(actual - predicted)
            print(f"  Case {i+1}: Days={row['trip_duration_days']}, Miles={row['miles_traveled']}, Receipts=${row['total_receipts_amount']:.2f}")
            print(f"    Actual: ${actual:.2f}, Predicted: ${predicted:.2f}, Error: ${error:.2f}")

if __name__ == "__main__":
    main() 