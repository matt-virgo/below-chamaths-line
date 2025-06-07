#!/usr/bin/env python3

from ultra_deep_learning import create_ultra_features, load_data
import pandas as pd

def analyze_ultraresnet_features():
    """Analyze the features used by the best-performing UltraResNet model"""
    
    # Load data and create features
    train_df, test_df = load_data()
    X_train = create_ultra_features(train_df)

    print('🎯 UltraResNet Model Features (Best Model - $58.91 MAE)')
    print('=' * 60)
    print(f'Total features: {len(X_train.columns)}')
    print()

    # List all features
    all_features = list(X_train.columns)
    
    # Group features by category
    print('📋 FEATURE BREAKDOWN:')
    print()

    # Basic derived
    basic = ['miles_per_day', 'receipts_per_day']
    basic_found = [f for f in basic if f in all_features]
    print(f'🔹 Basic Derived ({len(basic_found)}):')
    for f in basic_found:
        print(f'   {f}')

    # Core transformations
    core_patterns = ['total_trip_value', 'receipts_log', 'receipts_sqrt', 'receipts_squared', 'receipts_cubed', 
                    'miles_log', 'miles_sqrt', 'miles_squared', 'miles_cubed', 
                    'days_squared', 'days_cubed', 'days_fourth']
    core = [f for f in all_features if any(pattern in f for pattern in core_patterns)]
    print(f'\n🔹 Core Transformations ({len(core)}):')
    for f in core:
        print(f'   {f}')

    # Lucky cents
    lucky = [f for f in all_features if 'lucky' in f or 'cents' in f]
    print(f'\n🔹 Lucky Cents Pattern ({len(lucky)}):')
    for f in lucky:
        print(f'   {f}')

    # Interactions
    interaction_patterns = ['miles_receipts', 'days_receipts', 'days_miles', 'per_day_squared', 'miles_receipts_per_day']
    interactions = [f for f in all_features if any(pattern in f for pattern in interaction_patterns)]
    print(f'\n🔹 Interaction Features ({len(interactions)}):')
    for f in interactions:
        print(f'   {f}')

    # Ratios
    ratios = [f for f in all_features if 'ratio' in f or 'value_per_day' in f]
    print(f'\n🔹 Ratio Features ({len(ratios)}):')
    for f in ratios:
        print(f'   {f}')

    # Trigonometric
    trig = [f for f in all_features if 'sin' in f or 'cos' in f]
    print(f'\n🔹 Trigonometric Features ({len(trig)}):')
    for f in trig:
        print(f'   {f}')

    # Exponential
    exp = [f for f in all_features if 'exp' in f]
    print(f'\n🔹 Exponential Features ({len(exp)}):')
    for f in exp:
        print(f'   {f}')

    # High-order polynomial
    high_order_patterns = ['d2_m_r', 'd_m2_r', 'd_m_r2', 'sqrt_days', 'log_days']
    high_order = [f for f in all_features if any(pattern in f for pattern in high_order_patterns)]
    print(f'\n🔹 High-Order Polynomial ({len(high_order)}):')
    for f in high_order:
        print(f'   {f}')

    # Binned
    binned = [f for f in all_features if 'bin' in f]
    print(f'\n🔹 Binned Features ({len(binned)}):')
    for f in binned:
        print(f'   {f}')

    # Threshold indicators
    threshold_patterns = ['mpd_', 'rpd_', 'is_']
    threshold = [f for f in all_features if any(pattern in f for pattern in threshold_patterns)]
    print(f'\n🔹 Threshold Indicators ({len(threshold)}):')
    for f in threshold:
        print(f'   {f}')

    print()
    print('=' * 60)
    print(f'📊 TOTAL: {len(all_features)} features')
    
    # Show the original 3 features for comparison
    print()
    print('💡 Original Input Features (3):')
    print('   trip_duration_days')
    print('   miles_traveled') 
    print('   total_receipts_amount')
    
    print()
    print(f'🚀 Feature Expansion: 3 → {len(all_features)} features ({len(all_features)/3:.1f}x increase)')

if __name__ == "__main__":
    analyze_ultraresnet_features() 