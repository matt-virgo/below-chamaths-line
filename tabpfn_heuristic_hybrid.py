#!/usr/bin/env python3

"""
TabPFN + Heuristic Rules Hybrid Model
Combining TabPFN's foundation model predictions with domain-specific heuristic rules
derived from employee interviews and outlier analysis.
"""

import json
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

def create_v1_ultra_features(df):
    """Create V1's proven comprehensive feature set (58 features)"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
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
    
    # Lucky cents feature
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
    
    # Trigonometric features
    features_df['receipts_sin_1000'] = np.sin(R / 1000)
    features_df['receipts_cos_1000'] = np.cos(R / 1000)
    features_df['receipts_sin_500'] = np.sin(R / 500)
    features_df['receipts_cos_500'] = np.cos(R / 500)
    features_df['miles_sin_500'] = np.sin(M / 500)
    features_df['miles_cos_500'] = np.cos(M / 500)
    features_df['miles_sin_1000'] = np.sin(M / 1000)
    features_df['miles_cos_1000'] = np.cos(M / 1000)
    
    # Exponential features
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
    
    # Binned features
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
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in features_df.columns if col != 'reimbursement']
    
    return features_df[feature_cols]

def apply_interview_heuristics(df, base_predictions):
    """
    Apply heuristic rules derived from employee interviews and outlier analysis
    
    Key insights from interviews:
    - Kevin: "5-day trips with 180+ miles per day and under $100 per day in spending—that's a guaranteed bonus"
    - Kevin: "8+ day trips with high spending—that's a guaranteed penalty"  
    - Kevin: "180-220 miles per day sweet spot where the bonuses are maximized"
    - Lisa: "5-day trips almost always get a bonus"
    - Lisa: "really low receipts get penalized vs just taking per diem"
    - Multiple: Efficiency bonuses for high miles per day
    """
    
    adjusted_predictions = base_predictions.copy()
    adjustments_log = []
    
    # Extract basic variables
    D = df['trip_duration_days'].values
    M = df['miles_traveled'].values  
    R = df['total_receipts_amount'].values
    
    miles_per_day = M / D
    receipts_per_day = R / D
    
    for i in range(len(df)):
        days = D[i]
        miles = M[i]
        receipts = R[i]
        mpd = miles_per_day[i]
        rpd = receipts_per_day[i]
        
        adjustment_factor = 1.0
        reasons = []
        
        # 1. KEVIN'S SWEET SPOT COMBO: 5-day trips with high efficiency and modest spending
        if days == 5 and mpd >= 180 and rpd <= 100:
            adjustment_factor *= 1.08  # 8% bonus
            reasons.append("Kevin's sweet spot combo")
        
        # 2. 5-DAY BONUS (Lisa + multiple interviews)
        elif days == 5:
            adjustment_factor *= 1.05  # 5% bonus for 5-day trips
            reasons.append("5-day bonus")
        
        # 3. EFFICIENCY SWEET SPOT: 180-220 miles per day (Kevin)
        if 180 <= mpd <= 220:
            adjustment_factor *= 1.06  # 6% efficiency bonus
            reasons.append("Efficiency sweet spot")
        elif mpd > 300:  # Too much driving, system thinks you're not doing business
            adjustment_factor *= 0.95  # 5% penalty
            reasons.append("Excessive driving penalty")
        
        # 4. VACATION PENALTY: Long trips with high spending (Kevin)
        if days >= 8 and rpd > 120:
            adjustment_factor *= 0.92  # 8% penalty
            reasons.append("Vacation penalty")
        
        # 5. SMALL RECEIPT PENALTY: Very low receipts hurt vs no receipts (Dave + Lisa)
        if receipts < 50 and days > 1:  # Small receipts on multi-day trips
            adjustment_factor *= 0.94  # 6% penalty
            reasons.append("Small receipt penalty")
        
        # 6. ULTRA LOW ACTIVITY PENALTY: From outlier analysis
        if mpd < 50 and rpd < 30:  # Very low activity
            adjustment_factor *= 0.88  # 12% penalty
            reasons.append("Ultra low activity penalty")
        
        # 7. UNBALANCED TRIP PENALTIES: From outlier analysis
        receipts_to_miles_ratio = receipts / (miles + 1)
        if receipts_to_miles_ratio > 8:  # High receipts, low miles (conference trip?)
            adjustment_factor *= 0.96  # 4% penalty
            reasons.append("High receipts/low miles penalty")
        elif receipts_to_miles_ratio < 0.3 and miles > 500:  # High miles, very low receipts
            adjustment_factor *= 0.94  # 6% penalty  
            reasons.append("High miles/low receipts penalty")
        
        # 8. OPTIMAL SPENDING RANGES by trip length (Kevin)
        if days <= 3 and rpd > 75:  # Short trips should be modest
            adjustment_factor *= 0.97  # 3% penalty
            reasons.append("Short trip overspending")
        elif 4 <= days <= 6 and rpd > 120:  # Medium trips can spend more
            adjustment_factor *= 0.95  # 5% penalty
            reasons.append("Medium trip overspending")
        elif days >= 7 and rpd > 90:  # Long trips should be conservative
            adjustment_factor *= 0.93  # 7% penalty
            reasons.append("Long trip overspending")
        
        # 9. MEDIUM-HIGH RECEIPT BONUS: $600-800 gets good treatment (Lisa)
        if 600 <= receipts <= 800:
            adjustment_factor *= 1.04  # 4% bonus
            reasons.append("Medium-high receipt bonus")
        
        # 10. EXTREME DURATION PENALTIES: From outlier analysis
        if days == 1:  # Single day trips often problematic
            adjustment_factor *= 0.96  # 4% penalty
            reasons.append("Single day trip penalty")
        elif days >= 14:  # Very long trips
            adjustment_factor *= 0.94  # 6% penalty
            reasons.append("Very long trip penalty")
        
        # Apply the adjustment
        if adjustment_factor != 1.0:
            adjusted_predictions[i] *= adjustment_factor
            adjustments_log.append({
                'index': i,
                'days': days,
                'miles': miles,
                'receipts': receipts,
                'base_prediction': base_predictions[i],
                'adjusted_prediction': adjusted_predictions[i],
                'adjustment_factor': adjustment_factor,
                'reasons': reasons
            })
    
    return adjusted_predictions, adjustments_log

def get_tabpfn_predictions(X_train, y_train, X_test):
    """Get TabPFN predictions"""
    try:
        # Try to load existing results first
        tabpfn_results = pd.read_csv('tabpfn_v1_engineered_results.csv')
        return tabpfn_results['tabpfn_prediction'].values
    except FileNotFoundError:
        print("   🔄 Training TabPFN model...")
        
        try:
            from tabpfn import TabPFNRegressor
            
            # Train TabPFN
            tabpfn = TabPFNRegressor(device='cpu', N_ensemble_configurations=4)
            tabpfn.fit(X_train, y_train)
            predictions = tabpfn.predict(X_test)
            
            return predictions
        except Exception as e:
            print(f"   ❌ TabPFN failed: {str(e)}")
            return None

def main():
    print("🧠 TabPFN + Heuristic Rules Hybrid Model")
    print("="*70)
    print("Combining TabPFN foundation model with interview-derived business rules")
    print()
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data()
    
    print(f"Dataset overview:")
    print(f"   📊 Training samples: {len(train_df)}")
    print(f"   📊 Test samples: {len(test_df)}")
    
    # Create features
    print("\nCreating V1's comprehensive engineered features...")
    X_train = create_v1_ultra_features(train_df)
    X_test = create_v1_ultra_features(test_df)
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    print(f"✨ Using {X_train.shape[1]} V1 engineered features")
    
    # Get TabPFN base predictions
    print(f"\n{'='*70}")
    print(f"🤖 Getting TabPFN Base Predictions")
    print(f"{'='*70}")
    
    tabpfn_predictions = get_tabpfn_predictions(X_train.values, y_train, X_test.values)
    
    if tabpfn_predictions is None:
        print("❌ Could not get TabPFN predictions. Exiting.")
        return
    
    tabpfn_mae = mean_absolute_error(y_test, tabpfn_predictions)
    print(f"✅ TabPFN baseline: ${tabpfn_mae:.2f} MAE")
    
    # Apply heuristic rules
    print(f"\n{'='*70}")
    print(f"🎯 Applying Interview-Derived Heuristic Rules")
    print(f"{'='*70}")
    
    print("📋 Heuristic rules being applied:")
    print("   1. 🎯 Kevin's Sweet Spot: 5-day trips + 180+ mpd + <$100/day → +8% bonus")
    print("   2. 🎉 5-Day Bonus: All 5-day trips → +5% bonus") 
    print("   3. ⚡ Efficiency Sweet Spot: 180-220 miles/day → +6% bonus")
    print("   4. 🏖️ Vacation Penalty: 8+ days + >$120/day → -8% penalty")
    print("   5. 🧾 Small Receipt Penalty: <$50 receipts on multi-day → -6% penalty")
    print("   6. 📉 Ultra Low Activity: <50 mpd + <$30/day → -12% penalty")
    print("   7. ⚖️ Unbalanced Trip Penalties: High receipts/low miles or vice versa")
    print("   8. 💰 Spending Range Optimization: By trip length")
    print("   9. 💎 Medium Receipt Bonus: $600-800 receipts → +4% bonus")
    print("   10. ⏰ Extreme Duration Penalties: 1 day or 14+ days")
    
    hybrid_predictions, adjustments_log = apply_interview_heuristics(test_df, tabpfn_predictions)
    
    # Calculate performance
    hybrid_mae = mean_absolute_error(y_test, hybrid_predictions)
    hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_predictions))
    hybrid_r2 = r2_score(y_test, hybrid_predictions)
    
    # Compare results
    print(f"\n{'='*80}")
    print(f"🏆 HYBRID MODEL RESULTS")
    print(f"{'='*80}")
    
    improvement = tabpfn_mae - hybrid_mae
    improvement_pct = (improvement / tabpfn_mae) * 100
    
    print(f"📊 Performance Comparison:")
    print(f"   🤖 TabPFN Baseline:        ${tabpfn_mae:.2f} MAE")
    print(f"   🧠 TabPFN + Heuristics:    ${hybrid_mae:.2f} MAE")
    print(f"   📈 Improvement:            ${improvement:+.2f} ({improvement_pct:+.2f}%)")
    print(f"   📊 RMSE:                   ${hybrid_rmse:.2f}")
    print(f"   📊 R²:                     {hybrid_r2:.4f}")
    
    # Check if we beat the record
    previous_best = 55.96
    if hybrid_mae < previous_best:
        record_improvement = previous_best - hybrid_mae
        record_improvement_pct = (record_improvement / previous_best) * 100
        print(f"\n🎉 NEW WORLD RECORD!")
        print(f"   🏆 Previous best: ${previous_best:.2f} MAE")
        print(f"   🚀 New record:    ${hybrid_mae:.2f} MAE")
        print(f"   ⭐ Record improvement: ${record_improvement:.2f} ({record_improvement_pct:.2f}%)")
    else:
        record_gap = hybrid_mae - previous_best
        print(f"\n📊 vs Current Record (${previous_best:.2f}): ${record_gap:+.2f}")
    
    # Analyze adjustments
    print(f"\n📈 HEURISTIC ADJUSTMENTS ANALYSIS:")
    print(f"   Total test samples: {len(test_df)}")
    print(f"   Samples adjusted: {len(adjustments_log)}")
    print(f"   Adjustment rate: {len(adjustments_log)/len(test_df)*100:.1f}%")
    
    if len(adjustments_log) > 0:
        # Count adjustment types
        all_reasons = []
        for adj in adjustments_log:
            all_reasons.extend(adj['reasons'])
        
        from collections import Counter
        reason_counts = Counter(all_reasons)
        
        print(f"\n🎯 Most Common Adjustments:")
        for reason, count in reason_counts.most_common(10):
            print(f"   • {reason}: {count} samples")
        
        # Show some example adjustments
        print(f"\n💡 Example Adjustments:")
        for i, adj in enumerate(adjustments_log[:5]):
            direction = "↗️" if adj['adjustment_factor'] > 1 else "↘️"
            print(f"   {i+1}. {adj['days']}d, {adj['miles']:.0f}mi, ${adj['receipts']:.0f} "
                  f"{direction} ${adj['base_prediction']:.0f} → ${adj['adjusted_prediction']:.0f} "
                  f"({', '.join(adj['reasons'])})")
    
    # Save results
    results_df = pd.DataFrame({
        'actual': y_test,
        'tabpfn_prediction': tabpfn_predictions,
        'hybrid_prediction': hybrid_predictions,
        'tabpfn_error': np.abs(y_test - tabpfn_predictions),
        'hybrid_error': np.abs(y_test - hybrid_predictions),
        'improvement': np.abs(y_test - tabpfn_predictions) - np.abs(y_test - hybrid_predictions)
    })
    
    results_df.to_csv('tabpfn_heuristic_hybrid_results.csv', index=False)
    
    # Save adjustments log
    if adjustments_log:
        adjustments_df = pd.DataFrame(adjustments_log)
        adjustments_df.to_csv('heuristic_adjustments_log.csv', index=False)
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_heuristic_hybrid_results.csv")
        print(f"   📋 Adjustments: heuristic_adjustments_log.csv")
    
    # Final insights
    print(f"\n🧠 KEY INSIGHTS:")
    if improvement > 0:
        print(f"   ✅ Interview-derived heuristics improved TabPFN performance!")
        print(f"   🎯 Business domain knowledge adds value to foundation models")
        print(f"   📈 Human expertise + AI = better results")
    else:
        print(f"   🤔 Heuristics didn't improve overall performance")
        print(f"   💭 TabPFN may already capture these patterns implicitly")
        print(f"   🔬 Individual rules may help specific cases even if aggregate doesn't improve")
    
    print(f"   🔍 Most impactful rules help identify systematic biases")
    print(f"   ⚖️ Balance between human expertise and model predictions")
    print(f"   🚀 Foundation models can be enhanced with domain knowledge")

if __name__ == "__main__":
    main() 