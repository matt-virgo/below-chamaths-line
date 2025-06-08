#!/usr/bin/env python3

"""
TabPFN Business Rules Features
Testing TabPFN with sophisticated business-rules based feature engineering
incorporating employee interview insights (especially Kevin's domain expertise)
"""

import json
import math
import numpy as np
import pandas as pd
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

def engineer_features(df_input):
    """Business-rules based feature engineering incorporating interview insights"""
    df = df_input.copy()

    print("   🏢 Creating business-rules based features...")

    # Ensure trip_duration_days is at least 1 to avoid division by zero
    df['trip_duration_days_safe'] = df['trip_duration_days'].apply(lambda x: x if x > 0 else 1)

    # Base engineered features
    df['miles_per_day'] = df['miles_traveled'] / df['trip_duration_days_safe']
    df['receipts_per_day'] = df['total_receipts_amount'] / df['trip_duration_days_safe']
    
    df['receipt_cents_val'] = df['total_receipts_amount'].apply(
        lambda x: round((x - math.floor(x)) * 100) if isinstance(x, (int, float)) and not math.isnan(x) else 0
    )
    df['is_receipt_49_or_99_cents'] = df['receipt_cents_val'].apply(lambda x: 1 if x == 49 or x == 99 else 0).astype(int)
    
    # Trip length categories
    df['is_5_day_trip'] = (df['trip_duration_days'] == 5).astype(int)
    df['is_short_trip'] = (df['trip_duration_days'] < 4).astype(int)
    df['is_medium_trip'] = ((df['trip_duration_days'] >= 4) & (df['trip_duration_days'] <= 6)).astype(int)
    df['is_long_trip'] = ((df['trip_duration_days'] > 6) & (df['trip_duration_days'] < 8)).astype(int) # Specifically 7 days
    df['is_very_long_trip'] = (df['trip_duration_days'] >= 8).astype(int)

    # Polynomial features
    df['trip_duration_sq'] = df['trip_duration_days']**2
    df['miles_traveled_sq'] = df['miles_traveled']**2
    df['total_receipts_amount_sq'] = df['total_receipts_amount']**2
    df['miles_per_day_sq'] = df['miles_per_day']**2
    df['receipts_per_day_sq'] = df['receipts_per_day']**2

    # Mileage-based features
    df['miles_first_100'] = df['miles_traveled'].apply(lambda x: min(x, 100))
    df['miles_after_100'] = df['miles_traveled'].apply(lambda x: max(0, x - 100))
    df['is_high_mileage_trip'] = (df['miles_traveled'] > 500).astype(int)

    # Receipt-based features
    df['is_very_low_receipts_multiday'] = ((df['total_receipts_amount'] < 50) & (df['trip_duration_days'] > 1)).astype(int)
    df['is_moderate_receipts'] = ((df['total_receipts_amount'] >= 600) & (df['total_receipts_amount'] <= 800)).astype(int)
    df['is_high_receipts'] = ((df['total_receipts_amount'] > 800) & (df['total_receipts_amount'] <= 1200)).astype(int)
    df['is_very_high_receipts'] = (df['total_receipts_amount'] > 1200).astype(int)

    # Efficiency and Optimal Spending (Kevin's insights)
    df['is_optimal_miles_per_day_kevin'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
    
    def optimal_daily_spending(row):
        if row['is_short_trip']: # < 4 days
            return 1 if row['receipts_per_day'] < 75 else 0
        elif row['is_medium_trip']: # 4-6 days
            return 1 if row['receipts_per_day'] < 120 else 0
        # For 7 days (is_long_trip) or 8+ days (is_very_long_trip)
        elif row['is_long_trip'] or row['is_very_long_trip']: 
            return 1 if row['receipts_per_day'] < 90 else 0
        return 0 
    df['is_optimal_daily_spending_kevin'] = df.apply(optimal_daily_spending, axis=1).astype(int)

    # Interaction features
    df['duration_x_miles_per_day'] = df['trip_duration_days'] * df['miles_per_day']
    df['receipts_per_day_x_duration'] = df['receipts_per_day'] * df['trip_duration_days']
    
    df['interaction_kevin_sweet_spot'] = (df['is_5_day_trip'] & \
                                         (df['miles_per_day'] >= 180) & \
                                         (df['receipts_per_day'] < 100)).astype(int)
    
    df['interaction_kevin_vacation_penalty'] = (df['is_very_long_trip'] & \
                                               (df['receipts_per_day'] > 90)).astype(int)

    df['interaction_efficiency_metric'] = df['miles_traveled'] / (df['trip_duration_days_safe']**0.5 + 1e-6) 
    df['interaction_spending_mileage_ratio'] = df['total_receipts_amount'] / (df['miles_traveled'] + 1e-6)

    features_to_use = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount',
        'miles_per_day', 'receipts_per_day', 
        'is_receipt_49_or_99_cents',
        'is_5_day_trip', 'is_short_trip', 'is_medium_trip', 'is_long_trip', 'is_very_long_trip',
        'trip_duration_sq', 'miles_traveled_sq', 'total_receipts_amount_sq', 'miles_per_day_sq', 'receipts_per_day_sq',
        'miles_first_100', 'miles_after_100', 'is_high_mileage_trip',
        'is_very_low_receipts_multiday', 'is_moderate_receipts', 'is_high_receipts', 'is_very_high_receipts',
        'is_optimal_miles_per_day_kevin', 'is_optimal_daily_spending_kevin',
        'duration_x_miles_per_day', 'receipts_per_day_x_duration',
        'interaction_kevin_sweet_spot', 'interaction_kevin_vacation_penalty',
        'interaction_efficiency_metric', 'interaction_spending_mileage_ratio'
    ]
    
    print(f"   ✅ Business features created: {len(features_to_use)} features")
    return df[features_to_use]

def main():
    print("🚀 TabPFN Business Rules Features")
    print("="*70)
    print("Testing TabPFN with sophisticated business-rules based feature engineering")
    print("Incorporating employee interview insights (especially Kevin's domain expertise)")
    print()
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data()
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create business features
    print(f"\n{'='*70}")
    print(f"🏢 Creating Business-Rules Feature Set")
    print(f"{'='*70}")
    
    print("🎯 Feature categories:")
    print("   📊 Base Features: trip_duration_days, miles_traveled, total_receipts_amount")
    print("   🔄 Derived Features: miles_per_day, receipts_per_day")
    print("   💰 Lucky Cents: 49/99 cents pattern detection")
    print("   📅 Trip Categories: short/medium/long/very_long trip classification")
    print("   📈 Polynomial Features: squared transformations")
    print("   🛣️  Mileage Features: first 100 miles, excess miles, high mileage detection")
    print("   💵 Receipt Features: spending level categorization")
    print("   🎯 Kevin's Insights: optimal miles per day (180-220), daily spending thresholds")
    print("   🔗 Interaction Features: sweet spot detection, vacation penalty, efficiency metrics")
    
    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    total_features = X_train.shape[1]
    
    print(f"\n✨ BUSINESS FEATURE SET CREATED:")
    print(f"   📈 Total Features: {total_features}")
    print(f"   🏢 Business-focused approach with domain expertise")
    print(f"   🧠 Feature density: {total_features/len(train_df):.2f} features per sample")
    
    # Test TabPFN with business features
    print(f"\n{'='*70}")
    print(f"🤖 Training TabPFN with Business Rules Features")
    print(f"{'='*70}")
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("🚀 Initializing TabPFN...")
        print(f"   📊 Training on {len(X_train)} samples")
        print(f"   🏢 Using {total_features} business-engineered features")
        print("   🎯 Focus on domain expertise and employee insights")
        
        # Create TabPFN model
        tabpfn = TabPFNRegressor(device='cpu')
        
        print(f"🏋️ Training TabPFN...")
        
        # Convert to numpy arrays
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.astype(np.float32)
        
        tabpfn.fit(X_train_np, y_train_np)
        
        print(f"🔮 Generating predictions...")
        y_pred = tabpfn.predict(X_test_np)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*70}")
        print(f"🏆 TABPFN BUSINESS RULES RESULTS")
        print(f"{'='*70}")
        
        print(f"📊 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"   R-squared (R²): {r2:.4f}")
        
        # Compare to previous records
        previous_results = [
            ("TabPFN + Advanced Programmer V2 (Current Record)", 55.63),
            ("TabPFN Ultra Features (100+)", 55.96),
            ("TabPFN V1 Features", 55.96),
            ("TabPFN MEGA Features (176)", 57.11),
            ("V1 Neural Networks", 57.35),
            ("V4 Neural Networks", 59.76),
        ]
        
        print(f"\n📈 BUSINESS RULES COMPARISON:")
        print(f"   🆕 TabPFN Business Rules ({total_features}): ${mae:.2f} MAE")
        
        best_mae = 55.63
        improvement = best_mae - mae
        improvement_pct = (improvement / best_mae) * 100
        
        if mae < best_mae:
            print(f"   🎉 NEW WORLD RECORD! 🏆")
            print(f"   🥇 Previous best: ${best_mae:.2f} MAE")
            print(f"   📈 Improvement: ${improvement:.2f} ({improvement_pct:.2f}%)")
            print(f"   🏢 Business rules approach triumphs!")
        else:
            record_gap = mae - best_mae
            record_gap_pct = (record_gap / best_mae) * 100
            print(f"   📊 vs Current Record: ${record_gap:+.2f} ({record_gap_pct:+.2f}%)")
            if record_gap < 1.0:
                print(f"   🎯 Extremely close to record! Business approach very promising")
            elif record_gap < 2.0:
                print(f"   🎯 Very close to record! Business features showing strong potential")
            elif record_gap > 5.0:
                print(f"   ⚠️  Significant gap - may need more sophisticated features")
        
        for name, prev_mae in previous_results:
            diff = prev_mae - mae
            diff_pct = (diff / prev_mae) * 100
            emoji = "🎯" if mae < prev_mae else "📊"
            print(f"   {emoji} vs {name}: ${diff:+.2f} ({diff_pct:+.2f}%)")
        
        # Save results
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        results_df.to_csv('tabpfn_business_rules_results.csv', index=False)
        
        # Create comparison
        comparison_data = [{
            'model': f'TabPFN Business Rules ({total_features} features)',
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features': total_features,
            'notes': 'Business-focused feature engineering with employee interview insights (Kevin\'s domain expertise)'
        }]
        
        for name, prev_mae in previous_results:
            comparison_data.append({
                'model': name,
                'mae': prev_mae,
                'rmse': 'N/A',
                'r2': 'N/A',  
                'features': 'Various',
                'notes': 'Previous result for comparison'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('mae')
        comparison_df.to_csv('tabpfn_business_rules_comparison.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_business_rules_results.csv")
        print(f"   📈 Comparison: tabpfn_business_rules_comparison.csv")
        
        # Feature analysis
        print(f"\n🔍 BUSINESS FEATURE ANALYSIS:")
        
        # Check Kevin's insights activation
        kevin_sweet_spot_count = (X_train['interaction_kevin_sweet_spot'] == 1).sum()
        kevin_penalty_count = (X_train['interaction_kevin_vacation_penalty'] == 1).sum()
        optimal_miles_count = (X_train['is_optimal_miles_per_day_kevin'] == 1).sum()
        optimal_spending_count = (X_train['is_optimal_daily_spending_kevin'] == 1).sum()
        lucky_cents_count = (X_train['is_receipt_49_or_99_cents'] == 1).sum()
        
        print(f"   🎯 Kevin's Sweet Spot trips: {kevin_sweet_spot_count} ({kevin_sweet_spot_count/len(X_train)*100:.1f}%)")
        print(f"   ⚠️  Kevin's Vacation Penalty trips: {kevin_penalty_count} ({kevin_penalty_count/len(X_train)*100:.1f}%)")
        print(f"   🛣️  Optimal miles per day (180-220): {optimal_miles_count} ({optimal_miles_count/len(X_train)*100:.1f}%)")
        print(f"   💰 Optimal daily spending: {optimal_spending_count} ({optimal_spending_count/len(X_train)*100:.1f}%)")
        print(f"   🍀 Lucky cents (49/99): {lucky_cents_count} ({lucky_cents_count/len(X_train)*100:.1f}%)")
        
        # Trip distribution
        trip_dist = {
            'Short (<4 days)': (X_train['is_short_trip'] == 1).sum(),
            'Medium (4-6 days)': (X_train['is_medium_trip'] == 1).sum(),
            'Long (7 days)': (X_train['is_long_trip'] == 1).sum(),
            'Very Long (8+ days)': (X_train['is_very_long_trip'] == 1).sum(),
            '5-day trips': (X_train['is_5_day_trip'] == 1).sum()
        }
        
        print(f"\n   📅 Trip Duration Distribution:")
        for trip_type, count in trip_dist.items():
            print(f"      {trip_type}: {count} ({count/len(X_train)*100:.1f}%)")
        
        # Final insights
        print(f"\n🧠 BUSINESS RULES INSIGHTS:")
        if mae < best_mae:
            print(f"   🎉 BREAKTHROUGH! Business rules beat all previous approaches")
            print(f"   🏢 Domain expertise + employee interviews = winning combination")
            print(f"   🎯 Kevin's insights proved highly valuable")
            print(f"   📈 Focused feature engineering outperforms complex mathematics")
        elif mae < best_mae + 1.0:
            print(f"   🎯 Extremely competitive! Business approach nearly beat record")
            print(f"   🏢 Employee interview insights are highly valuable")
            print(f"   💡 Domain knowledge rivals complex feature engineering")
        else:
            print(f"   🤔 Good performance but didn't beat mathematical approaches")
            print(f"   📊 Business rules capture important patterns")
            print(f"   🔧 May benefit from combining with top mathematical features")
        
        print(f"\n🎯 KEY TAKEAWAYS:")
        print(f"   📊 Features tested: {total_features} (focused business approach)")
        print(f"   🏢 Approach: Employee interviews + domain expertise")
        print(f"   ⚡ TabPFN performance: {'Excellent' if mae < 60 else 'Good' if mae < 65 else 'Fair'}")
        print(f"   🧠 Business vs Mathematical: {'Business wins!' if mae < 55.96 else 'Mathematical still ahead' if mae > 57.35 else 'Very competitive'}")
        print(f"   🚀 Next steps: {'Celebrate!' if mae < best_mae else 'Consider hybrid approach' if mae < 58 else 'Refine business rules'}")
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 