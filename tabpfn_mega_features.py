#!/usr/bin/env python3

"""
TabPFN Mega Features - 200+ Feature Experiment
The ultimate feature engineering experiment pushing TabPFN to its limits
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.special import gamma, factorial
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

def sanitize_features(df):
    """Clean features by handling inf, -inf, and NaN values with aggressive measures"""
    print("   🔧 Sanitizing features for numerical stability...")
    
    # Convert to float32 to reduce precision issues
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].astype('float32')
    
    # Replace inf and -inf with finite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with 0
    df = df.fillna(0)
    
    # More aggressive clipping for extreme values
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32']:
            # Check for any remaining infinite values
            if np.any(np.isinf(df[col])) or np.any(np.isnan(df[col])):
                print(f"      ⚠️  Found infinite/NaN in {col}, fixing...")
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
            
            # Clip to safe ranges
            std_val = df[col].std()
            mean_val = df[col].mean()
            
            if std_val > 0:
                # Clip to within 10 standard deviations
                lower_bound = mean_val - 10 * std_val
                upper_bound = mean_val + 10 * std_val
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Final safety check - clip to reasonable absolute ranges
            df[col] = df[col].clip(-1e6, 1e6)
    
    # Final check for any problematic values
    inf_count = np.sum(np.isinf(df.values))
    nan_count = np.sum(np.isnan(df.values))
    
    if inf_count > 0 or nan_count > 0:
        print(f"      ⚠️  Still found {inf_count} inf and {nan_count} NaN values, forcing to 0...")
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"   ✅ Features sanitized - shape: {df.shape}")
    print(f"   📊 Value ranges: min={df.min().min():.2f}, max={df.max().max():.2f}")
    return df

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

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def fibonacci_ratio(n):
    """Calculate position in Fibonacci sequence or closest ratio"""
    if n <= 0:
        return 0
    
    # Generate Fibonacci numbers up to n
    fib = [1, 1]
    while fib[-1] < n:
        fib.append(fib[-1] + fib[-2])
    
    # Find closest Fibonacci number and calculate ratio
    closest_fib = min(fib, key=lambda x: abs(x - n))
    return n / closest_fib if closest_fib > 0 else 1

def create_v1_core_features(df):
    """Create V1's proven comprehensive feature set (40 core features)"""
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
    features_df['miles_sin_500'] = np.sin(M / 500)
    features_df['miles_cos_500'] = np.cos(M / 500)
    
    # Exponential features
    features_df['receipts_exp_norm'] = np.exp(R / 2000) - 1
    features_df['miles_exp_norm'] = np.exp(M / 1000) - 1
    
    # High-order interactions
    features_df['d2_m_r'] = (D ** 2) * M * R
    features_df['d_m2_r'] = D * (M ** 2) * R
    features_df['d_m_r2'] = D * M * (R ** 2)
    
    # Binned features
    features_df['receipts_bin_20'] = pd.cut(R, bins=20, labels=False)
    features_df['miles_bin_20'] = pd.cut(M, bins=20, labels=False)
    features_df['days_bin_10'] = pd.cut(D, bins=10, labels=False)
    
    # Special case indicators
    features_df['is_5_day_trip'] = (D == 5).astype(float)
    
    return features_df

def create_advanced_mathematical_features(df):
    """Create 35 advanced mathematical features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🧮 Creating advanced mathematical features...")
    
    # HIGHER ORDER POLYNOMIALS (10 features) - with clipping
    features_df['receipts_fifth'] = np.clip(R ** 5, 0, 1e12)
    features_df['miles_fifth'] = np.clip(M ** 5, 0, 1e12)
    features_df['days_fifth'] = np.clip(D ** 5, 0, 1e6)
    features_df['receipts_seventh'] = np.clip(R ** 7, 0, 1e15)
    features_df['miles_seventh'] = np.clip(M ** 7, 0, 1e15)
    features_df['receipts_eighth'] = np.clip(R ** 8, 0, 1e16)
    features_df['receipts_ninth'] = np.clip(R ** 9, 0, 1e18)
    features_df['receipts_tenth'] = np.clip(R ** 10, 0, 1e20)
    features_df['mixed_power_5_3_2'] = np.clip((R ** 5) * (M ** 3) * (D ** 2), 0, 1e15)
    features_df['mixed_power_4_4_4'] = np.clip((R ** 4) * (M ** 4) * (D ** 4), 0, 1e15)
    
    # ADVANCED TRIGONOMETRIC (10 features) - with safe scaling
    features_df['receipts_tan_1000'] = np.clip(np.tan(R / 1000), -10, 10)
    features_df['miles_tan_500'] = np.clip(np.tan(M / 500), -10, 10)
    features_df['receipts_sin_250'] = np.sin(R / 250)
    features_df['receipts_cos_250'] = np.cos(R / 250)
    features_df['miles_sin_250'] = np.sin(M / 250)
    features_df['receipts_sec'] = np.clip(1 / (np.cos(R / 1000) + 1e-8), -100, 100)
    features_df['receipts_csc'] = np.clip(1 / (np.sin(R / 1000) + 1e-8), -100, 100)
    features_df['receipts_cot'] = np.clip(1 / (np.tan(R / 1000) + 1e-8), -100, 100)
    features_df['phase_shift_receipts'] = np.sin(R / 500 + np.pi/4)
    features_df['amplitude_mod'] = np.sin(R / 1000) * np.cos(M / 500)
    
    # HYPERBOLIC FUNCTIONS (5 features) - with safe scaling
    features_df['receipts_sinh'] = np.clip(np.sinh(R / 5000), -100, 100)
    features_df['receipts_cosh'] = np.clip(np.cosh(R / 5000), 0, 100)
    features_df['miles_tanh'] = np.tanh(M / 1000)
    features_df['receipts_coth'] = np.clip(1 / (np.tanh(R / 5000) + 1e-8), -100, 100)
    features_df['hyperbolic_product'] = np.clip(np.sinh(R / 5000) * np.cosh(M / 2500), -100, 100)
    
    # LOGARITHMIC VARIANTS (5 features)
    features_df['receipts_log10'] = np.log10(R + 1)
    features_df['miles_log10'] = np.log10(M + 1)
    features_df['receipts_log2'] = np.log2(R + 1)
    features_df['natural_log_product'] = np.log1p(R) * np.log1p(M)
    features_df['log_harmonic_mean'] = np.log1p(2 / (1/(R+1) + 1/(M+1)))
    
    # RECIPROCAL AND INVERSE (5 features)
    features_df['receipts_reciprocal'] = 1 / (R + 1)
    features_df['miles_reciprocal'] = 1 / (M + 1)
    features_df['days_reciprocal'] = 1 / (D + 1)
    features_df['geometric_reciprocal'] = 1 / (np.sqrt(R * M) + 1)
    features_df['harmonic_reciprocal'] = (1/(R+1) + 1/(M+1) + 1/(D+1)) / 3
    
    return features_df

def create_statistical_distribution_features(df):
    """Create 25 statistical and distribution features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   📊 Creating statistical distribution features...")
    
    # PERCENTILE AND RANKING FEATURES (8 features)
    all_receipts = np.concatenate([R.values] * 5)
    all_miles = np.concatenate([M.values] * 5)
    features_df['receipts_percentile'] = R.apply(lambda x: stats.percentileofscore(all_receipts, x) / 100)
    features_df['miles_percentile'] = M.apply(lambda x: stats.percentileofscore(all_miles, x) / 100)
    features_df['receipts_zscore'] = (R - R.mean()) / (R.std() + 1e-8)
    features_df['miles_zscore'] = (M - M.mean()) / (M.std() + 1e-8)
    features_df['receipts_rank'] = R.rank(pct=True)
    features_df['miles_rank'] = M.rank(pct=True)
    features_df['combined_percentile'] = (features_df['receipts_percentile'] + features_df['miles_percentile']) / 2
    features_df['percentile_spread'] = np.abs(features_df['receipts_percentile'] - features_df['miles_percentile'])
    
    # DISTRIBUTION MOMENT FEATURES (6 features) - with clipping
    features_df['receipts_skewness'] = np.clip(((R - R.mean()) / (R.std() + 1e-8)) ** 3, -10, 10)
    features_df['miles_skewness'] = np.clip(((M - M.mean()) / (M.std() + 1e-8)) ** 3, -10, 10)
    features_df['receipts_kurtosis'] = np.clip(((R - R.mean()) / (R.std() + 1e-8)) ** 4, 0, 100)
    features_df['miles_kurtosis'] = np.clip(((M - M.mean()) / (M.std() + 1e-8)) ** 4, 0, 100)
    features_df['cross_moment_3'] = np.clip(((R - R.mean()) * (M - M.mean()) ** 2) / ((R.std() * M.std()) + 1e-8), -100, 100)
    features_df['cross_moment_4'] = np.clip(((R - R.mean()) ** 2 * (M - M.mean()) ** 2) / ((R.std() ** 2 * M.std() ** 2) + 1e-8), 0, 1000)
    
    # PROBABILITY DENSITY APPROXIMATIONS (6 features) - with error handling
    try:
        features_df['receipts_normal_density'] = stats.norm.pdf(R, R.mean(), R.std() + 1e-8)
        features_df['miles_normal_density'] = stats.norm.pdf(M, M.mean(), M.std() + 1e-8)
        features_df['receipts_lognormal_density'] = stats.lognorm.pdf(R, s=1, scale=np.exp(np.log(R.mean() + 1)))
        features_df['receipts_gamma_density'] = stats.gamma.pdf(R, a=2, scale=R.mean()/2 + 1e-8)
        features_df['receipts_beta_scaled'] = stats.beta.pdf(np.clip(R / (R.max() + 1), 0, 1), a=2, b=2)
        features_df['joint_density_approx'] = features_df['receipts_normal_density'] * features_df['miles_normal_density']
    except:
        # Fallback to simple features if density calculations fail
        features_df['receipts_normal_density'] = np.exp(-0.5 * ((R - R.mean()) / (R.std() + 1e-8))**2)
        features_df['miles_normal_density'] = np.exp(-0.5 * ((M - M.mean()) / (M.std() + 1e-8))**2)
        features_df['receipts_lognormal_density'] = 1 / (R + 1)
        features_df['receipts_gamma_density'] = np.exp(-R / R.mean())
        features_df['receipts_beta_scaled'] = (R / (R.max() + 1)) * (1 - R / (R.max() + 1))
        features_df['joint_density_approx'] = features_df['receipts_normal_density'] * features_df['miles_normal_density']
    
    # ENTROPY AND INFORMATION FEATURES (5 features)
    receipts_bins = pd.cut(R, bins=10, labels=False)
    miles_bins = pd.cut(M, bins=10, labels=False)
    receipts_probs = receipts_bins.value_counts(normalize=True) + 1e-8
    miles_probs = miles_bins.value_counts(normalize=True) + 1e-8
    features_df['receipts_entropy'] = -np.sum(receipts_probs * np.log2(receipts_probs))
    features_df['miles_entropy'] = -np.sum(miles_probs * np.log2(miles_probs))
    features_df['mutual_information_approx'] = features_df['receipts_entropy'] + features_df['miles_entropy']
    features_df['information_gain'] = np.log2(R + 1) - np.log2(R.mean() + 1)
    features_df['relative_entropy'] = np.log2((R + 1) / (R.mean() + 1))
    
    return features_df

def create_number_theory_features(df):
    """Create 20 number theory and combinatorial features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🔢 Creating number theory features...")
    
    # PRIME AND DIVISIBILITY FEATURES (8 features)
    features_df['receipts_prime_distance'] = R.apply(lambda x: min([abs(x - p) for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if abs(x - p) < 100], default=100))
    features_df['miles_prime_distance'] = M.apply(lambda x: min([abs(x - p) for p in [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47] if abs(x - p) < 100], default=100))
    features_df['receipts_divisor_count'] = R.astype(int).apply(lambda x: len([i for i in range(1, min(x+1, 100)) if x % i == 0]))
    features_df['miles_divisor_count'] = M.astype(int).apply(lambda x: len([i for i in range(1, min(x+1, 100)) if x % i == 0]))
    features_df['gcd_receipts_miles'] = np.gcd(R.astype(int), M.astype(int))
    features_df['lcm_receipts_miles'] = np.lcm(R.astype(int), M.astype(int)) / 1000  # Scaled
    features_df['receipts_digital_root'] = R.astype(int).apply(lambda x: x if x < 10 else sum(int(d) for d in str(x)) % 9 or 9)
    features_df['miles_digital_root'] = M.astype(int).apply(lambda x: x if x < 10 else sum(int(d) for d in str(x)) % 9 or 9)
    
    # FIBONACCI AND SEQUENCE FEATURES (6 features)
    features_df['fib_ratio_receipts'] = R.apply(fibonacci_ratio)
    features_df['fib_ratio_miles'] = M.apply(fibonacci_ratio)
    features_df['golden_ratio_receipts'] = np.abs(R - R * 1.618) / (R + 1)
    features_df['tribonacci_distance'] = R.apply(lambda x: min(abs(x - t) for t in [1,1,2,4,7,13,24,44,81,149]) if x < 200 else 200)
    features_df['lucas_distance'] = R.apply(lambda x: min(abs(x - l) for l in [2,1,3,4,7,11,18,29,47,76]) if x < 100 else 100)
    features_df['catalan_distance'] = R.apply(lambda x: min(abs(x - c) for c in [1,1,2,5,14,42,132]) if x < 150 else 150)
    
    # MODULAR ARITHMETIC PATTERNS (6 features)
    features_df['mod_pattern_7'] = ((R * M * D) % 7) / 7
    features_df['mod_pattern_11'] = ((R + M + D) % 11) / 11
    features_df['mod_pattern_13'] = ((R * M + D) % 13) / 13
    features_df['quadratic_residue_7'] = ((R.astype(int) ** 2) % 7) / 7
    features_df['cubic_residue_11'] = ((R.astype(int) ** 3) % 11) / 11
    features_df['mixed_modular'] = ((R.astype(int) * 31 + M.astype(int) * 17 + D * 13) % 997) / 997
    
    return features_df

def create_spectral_signal_features(df):
    """Create 15 spectral analysis and signal processing features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   📡 Creating spectral analysis features...")
    
    # FOURIER TRANSFORM FEATURES (8 features)
    # Create synthetic time series from our features
    receipts_signal = np.array([R.iloc[i % len(R)] for i in range(64)])  # Create 64-point signal
    miles_signal = np.array([M.iloc[i % len(M)] for i in range(64)])
    
    # Compute FFT
    receipts_fft = np.abs(fft(receipts_signal))
    miles_fft = np.abs(fft(miles_signal))
    
    # Extract spectral features for each sample
    features_df['receipts_spectral_centroid'] = R.apply(lambda x: np.mean(receipts_fft * np.arange(len(receipts_fft))))
    features_df['receipts_spectral_bandwidth'] = R.apply(lambda x: np.sqrt(np.mean((receipts_fft - np.mean(receipts_fft))**2)))
    features_df['receipts_spectral_rolloff'] = R.apply(lambda x: np.where(np.cumsum(receipts_fft) >= 0.85 * np.sum(receipts_fft))[0][0] if len(np.where(np.cumsum(receipts_fft) >= 0.85 * np.sum(receipts_fft))[0]) > 0 else 32)
    features_df['miles_spectral_centroid'] = M.apply(lambda x: np.mean(miles_fft * np.arange(len(miles_fft))))
    features_df['spectral_contrast'] = features_df['receipts_spectral_centroid'] - features_df['miles_spectral_centroid']
    features_df['dominant_frequency'] = R.apply(lambda x: np.argmax(receipts_fft))
    features_df['spectral_energy'] = R.apply(lambda x: np.sum(receipts_fft ** 2))
    features_df['spectral_entropy_approx'] = R.apply(lambda x: -np.sum((receipts_fft / np.sum(receipts_fft)) * np.log2(receipts_fft / np.sum(receipts_fft) + 1e-8)))
    
    # WAVELET-INSPIRED FEATURES (4 features)
    features_df['receipts_high_freq'] = np.abs(R - R.rolling(window=3, center=True).mean().fillna(R))
    features_df['receipts_low_freq'] = R.rolling(window=5, center=True).mean().fillna(R)
    features_df['miles_high_freq'] = np.abs(M - M.rolling(window=3, center=True).mean().fillna(M))
    features_df['frequency_ratio'] = features_df['receipts_high_freq'] / (features_df['receipts_low_freq'] + 1)
    
    # AUTOCORRELATION FEATURES (3 features)
    features_df['receipts_autocorr_lag1'] = R.rolling(window=10).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0).fillna(0)
    features_df['miles_autocorr_lag1'] = M.rolling(window=10).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0).fillna(0)
    features_df['cross_correlation'] = features_df['receipts_autocorr_lag1'] * features_df['miles_autocorr_lag1']
    
    return features_df

def create_chaos_complexity_features(df):
    """Create 15 chaos theory and complexity features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🌀 Creating chaos theory features...")
    
    # LOGISTIC MAP FEATURES (5 features) - with safe modulo
    features_df['logistic_map_r'] = R.apply(lambda x: 4 * (x % 1) * (1 - (x % 1)))
    features_df['logistic_map_m'] = M.apply(lambda x: 3.8 * (x % 1) * (1 - (x % 1)))
    features_df['tent_map_r'] = R.apply(lambda x: 2 * min(x % 1, 1 - (x % 1)))
    features_df['henon_x'] = R.apply(lambda x: np.clip(1 - 1.4 * ((x % 1) ** 2) + 0.3 * (x % 1), -10, 10))
    features_df['sine_map'] = R.apply(lambda x: np.sin(np.pi * (x % 1)))
    
    # FRACTAL DIMENSION APPROXIMATIONS (5 features) - with safe calculations
    features_df['box_counting_r'] = R.apply(lambda x: np.clip(np.log(max(1, len(set([int(x * i / 10) for i in range(10)])))) / np.log(10), 0, 3))
    features_df['correlation_dim_approx'] = R.apply(lambda x: np.clip(np.log(max(1, x % 100)) / np.log(max(1, x % 10)), 0, 10))
    features_df['hausdorff_approx'] = R.apply(lambda x: np.clip(np.log(max(1, len(str(int(x))))) / np.log(2), 0, 10))
    features_df['minkowski_dim'] = M.apply(lambda x: np.clip(1 + np.log(max(1, x % 100)) / np.log(max(1, x % 10)), 0, 10))
    features_df['fractal_similarity'] = np.abs(features_df['box_counting_r'] - features_df['correlation_dim_approx'])
    
    # LYAPUNOV EXPONENT APPROXIMATIONS (3 features) - with safe log
    features_df['lyapunov_approx_r'] = R.apply(lambda x: np.clip(np.log(max(1e-8, abs(1 - 2 * (x % 1)))), -10, 10))
    features_df['lyapunov_approx_m'] = M.apply(lambda x: np.clip(np.log(max(1e-8, abs(2 - 4 * (x % 1)))), -10, 10))
    features_df['stability_measure'] = features_df['lyapunov_approx_r'] + features_df['lyapunov_approx_m']
    
    # COMPLEXITY MEASURES (2 features)
    features_df['kolmogorov_approx'] = R.apply(lambda x: len(str(int(x)).replace('0', '').replace('1', '').replace('2', '')))
    features_df['lempel_ziv_approx'] = R.apply(lambda x: len(set(str(int(x))[i:i+2] for i in range(max(0, len(str(int(x)))-1)))))
    
    return features_df

def create_graph_network_features(df):
    """Create 10 graph theory and network features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   🕸️  Creating graph theory features...")
    
    # ADJACENCY AND CONNECTIVITY (4 features)
    features_df['node_degree'] = (R > R.median()).astype(int) + (M > M.median()).astype(int) + (D > D.median()).astype(int)
    features_df['clustering_coeff'] = features_df['node_degree'] / 3
    features_df['betweenness_approx'] = R.apply(lambda x: (x - R.min()) / (R.max() - R.min()) if R.max() != R.min() else 0)
    features_df['centrality_measure'] = (features_df['betweenness_approx'] + features_df['clustering_coeff']) / 2
    
    # DISTANCE METRICS (3 features)
    features_df['manhattan_distance'] = np.abs(R - R.mean()) + np.abs(M - M.mean()) + np.abs(D - D.mean())
    features_df['euclidean_distance'] = np.sqrt((R - R.mean())**2 + (M - M.mean())**2 + (D - D.mean())**2)
    features_df['cosine_similarity'] = (R * M * D) / (np.sqrt(R**2 + M**2 + D**2) + 1)
    
    # NETWORK FLOW (3 features)
    features_df['flow_capacity'] = R / (D + 1)  # Like flow per time unit
    features_df['network_efficiency'] = M / (R + D + 1)  # Output per input
    features_df['bottleneck_measure'] = np.minimum(R/100, np.minimum(M/100, D))
    
    return features_df

def create_temporal_cyclical_features(df):
    """Create 15 temporal and cyclical pattern features"""
    features_df = df.copy()
    
    # Basic variables
    D = df['trip_duration_days']
    M = df['miles_traveled']
    R = df['total_receipts_amount']
    
    print("   ⏰ Creating temporal cyclical features...")
    
    # CYCLICAL PATTERNS (8 features)
    features_df['weekly_cycle_r'] = np.sin(2 * np.pi * R / 7)
    features_df['weekly_cycle_m'] = np.cos(2 * np.pi * M / 7)
    features_df['monthly_cycle_r'] = np.sin(2 * np.pi * R / 30)
    features_df['monthly_cycle_m'] = np.cos(2 * np.pi * M / 30)
    features_df['quarterly_cycle'] = np.sin(2 * np.pi * R / 90)
    features_df['daily_pattern'] = np.sin(2 * np.pi * D / 24)
    features_df['seasonal_strength'] = np.sqrt(features_df['weekly_cycle_r']**2 + features_df['monthly_cycle_r']**2)
    features_df['phase_alignment'] = np.abs(features_df['weekly_cycle_r'] - features_df['weekly_cycle_m'])
    
    # TREND FEATURES (4 features)
    features_df['linear_trend'] = np.arange(len(df)) * (R.iloc[-1] - R.iloc[0]) / len(df) if len(df) > 1 else 0
    features_df['momentum'] = R - R.rolling(window=5, min_periods=1).mean()
    features_df['acceleration'] = features_df['momentum'] - features_df['momentum'].rolling(window=3, min_periods=1).mean()
    features_df['trend_strength'] = np.abs(features_df['linear_trend']) / (R.std() + 1)
    
    # PERSISTENCE FEATURES (3 features)
    features_df['hurst_approx'] = R.rolling(window=10, min_periods=1).apply(lambda x: np.log(np.std(x) + 1e-8) / np.log(len(x)) if len(x) > 1 else 0.5)
    features_df['persistence_r'] = (R > R.shift(1)).astype(int).rolling(window=5, min_periods=1).sum() / 5
    features_df['persistence_m'] = (M > M.shift(1)).astype(int).rolling(window=5, min_periods=1).sum() / 5
    
    return features_df

def create_mega_feature_set(df):
    """Combine all feature categories for 200+ total features"""
    print("🚀 Creating MEGA feature set with 8 comprehensive categories...")
    
    # Start with V1 core features (40)
    print("   📊 V1 Core Features: ~40")
    combined_df = create_v1_core_features(df)
    
    # Add advanced mathematical features (35)
    print("   🧮 Advanced Mathematical: ~35")
    math_df = create_advanced_mathematical_features(df)
    for col in math_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = math_df[col]
    
    # Add statistical distribution features (25)
    print("   📊 Statistical Distribution: ~25")
    stats_df = create_statistical_distribution_features(df)
    for col in stats_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = stats_df[col]
    
    # Add number theory features (20)
    print("   🔢 Number Theory: ~20")
    number_df = create_number_theory_features(df)
    for col in number_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = number_df[col]
    
    # Add spectral signal features (15)
    print("   📡 Spectral Analysis: ~15")
    spectral_df = create_spectral_signal_features(df)
    for col in spectral_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = spectral_df[col]
    
    # Add chaos complexity features (15)
    print("   🌀 Chaos Theory: ~15")
    chaos_df = create_chaos_complexity_features(df)
    for col in chaos_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = chaos_df[col]
    
    # Add graph network features (10)
    print("   🕸️  Graph Theory: ~10")
    graph_df = create_graph_network_features(df)
    for col in graph_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = graph_df[col]
    
    # Add temporal cyclical features (15)
    print("   ⏰ Temporal Cyclical: ~15")
    temporal_df = create_temporal_cyclical_features(df)
    for col in temporal_df.columns:
        if col not in combined_df.columns:
            combined_df[col] = temporal_df[col]
    
    # Get feature columns (exclude target)
    feature_cols = [col for col in combined_df.columns if col != 'reimbursement']
    
    # Apply sanitization to ensure numerical stability
    feature_df = combined_df[feature_cols]
    feature_df = sanitize_features(feature_df)
    
    return feature_df

def debug_feature_problems(X_train, X_test):
    """Debug and identify problematic features"""
    print("   🔍 Debugging feature problems...")
    
    # Check for problematic values in training set
    train_problems = []
    for i, col in enumerate(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1])):
        col_data = X_train.iloc[:, i] if hasattr(X_train, 'iloc') else X_train[:, i]
        
        inf_count = np.sum(np.isinf(col_data))
        nan_count = np.sum(np.isnan(col_data))
        large_count = np.sum(np.abs(col_data) > 1e10)
        
        if inf_count > 0 or nan_count > 0 or large_count > 0:
            train_problems.append(f"Feature {col}: {inf_count} inf, {nan_count} nan, {large_count} large")
    
    # Check for problematic values in test set
    test_problems = []
    for i, col in enumerate(X_test.columns if hasattr(X_test, 'columns') else range(X_test.shape[1])):
        col_data = X_test.iloc[:, i] if hasattr(X_test, 'iloc') else X_test[:, i]
        
        inf_count = np.sum(np.isinf(col_data))
        nan_count = np.sum(np.isnan(col_data))
        large_count = np.sum(np.abs(col_data) > 1e10)
        
        if inf_count > 0 or nan_count > 0 or large_count > 0:
            test_problems.append(f"Feature {col}: {inf_count} inf, {nan_count} nan, {large_count} large")
    
    if train_problems:
        print(f"   ⚠️  Training set problems:")
        for problem in train_problems[:5]:  # Show first 5
            print(f"      {problem}")
        if len(train_problems) > 5:
            print(f"      ... and {len(train_problems) - 5} more")
    
    if test_problems:
        print(f"   ⚠️  Test set problems:")
        for problem in test_problems[:5]:  # Show first 5
            print(f"      {problem}")
        if len(test_problems) > 5:
            print(f"      ... and {len(test_problems) - 5} more")
    
    if not train_problems and not test_problems:
        print("   ✅ No obvious feature problems detected")
        return True
    else:
        print(f"   ❌ Found {len(train_problems)} train and {len(test_problems)} test feature problems")
        return False

def main():
    print("🚀 TabPFN MEGA Features - 200+ Feature Experiment")
    print("="*80)
    print("The ULTIMATE feature engineering experiment - pushing TabPFN to its absolute limits!")
    print()
    
    # Load data
    print("📊 Loading data...")
    train_df, test_df = load_data()
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Create mega feature set
    print(f"\n{'='*80}")
    print(f"🔧 Creating MEGA-Comprehensive Feature Set")
    print(f"{'='*80}")
    
    print("🎯 Feature categories breakdown:")
    print("   📊 V1 Core Features: ~40 (proven best baseline)")
    print("   🧮 Advanced Mathematical: ~35 (polynomials, trig, hyperbolic)")
    print("   📊 Statistical Distribution: ~25 (percentiles, moments, entropy)")
    print("   🔢 Number Theory: ~20 (primes, Fibonacci, modular arithmetic)")
    print("   📡 Spectral Analysis: ~15 (FFT, wavelets, autocorrelation)")
    print("   🌀 Chaos Theory: ~15 (logistic maps, fractals, Lyapunov)")
    print("   🕸️  Graph Theory: ~10 (networks, distances, centrality)")
    print("   ⏰ Temporal Cyclical: ~15 (cycles, trends, persistence)")
    print("   🎯 TARGET: 200+ total features")
    
    X_train = create_mega_feature_set(train_df)
    X_test = create_mega_feature_set(test_df)
    
    y_train = train_df['reimbursement'].values
    y_test = test_df['reimbursement'].values
    
    total_features = X_train.shape[1]
    
    print(f"\n✨ MEGA FEATURE SET CREATED:")
    print(f"   📈 Total Features: {total_features}")
    print(f"   🎯 Target Achieved: {'YES! 🎉' if total_features >= 200 else f'Close! ({total_features}/200)'}")
    print(f"   🧠 Feature density: {total_features/len(train_df):.2f} features per sample")
    
    # Debug feature problems
    print(f"\n{'='*80}")
    print(f"🔍 Debugging Feature Quality")
    print(f"{'='*80}")
    
    features_ok = debug_feature_problems(X_train, X_test)
    
    if not features_ok:
        print("\n⚠️  Detected feature problems - applying fallback strategy...")
        # Use only V1 core features as fallback
        print("   🔄 Falling back to V1 core features only...")
        X_train = create_v1_core_features(train_df)
        X_test = create_v1_core_features(test_df)
        
        # Remove target column
        feature_cols = [col for col in X_train.columns if col != 'reimbursement']
        X_train = X_train[feature_cols]
        X_test = X_test[feature_cols]
        
        # Re-sanitize
        X_train = sanitize_features(X_train)
        X_test = sanitize_features(X_test)
        
        total_features = X_train.shape[1]
        print(f"   📊 Fallback feature count: {total_features}")
        
        # Re-check
        features_ok = debug_feature_problems(X_train, X_test)
        if not features_ok:
            print("   ❌ Even core features have problems - this needs investigation")
            return
    
    # Test TabPFN with mega features
    print(f"\n{'='*80}")
    print(f"🤖 Training TabPFN with {'MEGA' if total_features > 100 else 'Core'} Features")
    print(f"{'='*80}")
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("🚀 Initializing TabPFN...")
        print(f"   📊 Training on {len(X_train)} samples")
        print(f"   🔧 Using {total_features} engineered features")
        if total_features > 100:
            print("   ⚡ This will take significant time with 100+ features...")
            print("   🧠 TabPFN will need to process a high-dimensional space")
        
        # Create TabPFN model
        tabpfn = TabPFNRegressor(device='cpu')
        
        print(f"🏋️ Training TabPFN...")
        
        # Convert to numpy arrays with explicit float32
        X_train_np = X_train.values.astype(np.float32)
        X_test_np = X_test.values.astype(np.float32)
        y_train_np = y_train.astype(np.float32)
        
        # Final safety check on the numpy arrays
        if np.any(np.isinf(X_train_np)) or np.any(np.isnan(X_train_np)):
            print("   ⚠️  Still have problems in training data, final cleanup...")
            X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isinf(X_test_np)) or np.any(np.isnan(X_test_np)):
            print("   ⚠️  Still have problems in test data, final cleanup...")
            X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        tabpfn.fit(X_train_np, y_train_np)
        
        print(f"🔮 Generating predictions...")
        y_pred = tabpfn.predict(X_test_np)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{'='*80}")
        print(f"🏆 TABPFN {'MEGA' if total_features > 100 else 'CORE'} FEATURES RESULTS")
        print(f"{'='*80}")
        
        print(f"📊 Performance Metrics:")
        print(f"   Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"   Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"   R-squared (R²): {r2:.4f}")
        
        # Compare to previous records
        previous_results = [
            ("TabPFN + Advanced Programmer V2 (Current Record)", 55.63),
            ("TabPFN Ultra Features (100+)", 55.96),
            ("TabPFN V1 Features", 55.96),
            ("V1 Neural Networks", 57.35),
            ("V4 Neural Networks", 59.76),
        ]
        
        print(f"\n📈 {'MEGA' if total_features > 100 else 'CORE'} FEATURES COMPARISON:")
        print(f"   🆕 TabPFN Features ({total_features}): ${mae:.2f} MAE")
        
        best_mae = 55.63
        improvement = best_mae - mae
        improvement_pct = (improvement / best_mae) * 100
        
        if mae < best_mae:
            print(f"   🎉 NEW WORLD RECORD! 🏆")
            print(f"   🥇 Previous best: ${best_mae:.2f} MAE")
            print(f"   📈 Improvement: ${improvement:.2f} ({improvement_pct:.2f}%)")
            if total_features > 100:
                print(f"   🚀 MEGA features breakthrough achieved!")
            else:
                print(f"   💪 Even with fallback features, we beat the record!")
        else:
            record_gap = mae - best_mae
            record_gap_pct = (record_gap / best_mae) * 100
            print(f"   📊 vs Current Record: ${record_gap:+.2f} ({record_gap_pct:+.2f}%)")
            if record_gap < 2.0:
                print(f"   🎯 Very close to record! Features showing promise")
            elif record_gap > 5.0:
                print(f"   ⚠️  Significant gap - possible feature issues or saturation")
        
        for name, prev_mae in previous_results:
            diff = prev_mae - mae
            diff_pct = (diff / prev_mae) * 100
            emoji = "🎯" if mae < prev_mae else "📊"
            print(f"   {emoji} vs {name}: ${diff:+.2f} ({diff_pct:+.2f}%)")
        
        # Save results
        filename_suffix = "mega" if total_features > 100 else "core_fallback"
        results_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'absolute_error': np.abs(y_test - y_pred)
        })
        
        results_df.to_csv(f'tabpfn_{filename_suffix}_features_results.csv', index=False)
        
        # Create comparison
        model_name = f"TabPFN {'MEGA' if total_features > 100 else 'Core Fallback'} Features ({total_features} features)"
        comparison_data = [{
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'features': total_features,
            'notes': f"{'8 comprehensive categories' if total_features > 100 else 'V1 core features only - fallback due to numerical issues'}"
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
        comparison_df.to_csv(f'tabpfn_{filename_suffix}_features_comparison.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   📊 Predictions: tabpfn_{filename_suffix}_features_results.csv")
        print(f"   📈 Comparison: tabpfn_{filename_suffix}_features_comparison.csv")
        
        # Final insights
        print(f"\n🧠 {'MEGA' if total_features > 100 else 'FALLBACK'} FEATURES ANALYSIS:")
        if mae < best_mae:
            if total_features > 100:
                print(f"   🎉 BREAKTHROUGH! {total_features} mega features achieved new record")
                print(f"   🏆 MEGA feature engineering proves its worth")
                print(f"   📈 TabPFN scales beautifully to high dimensions")
                print(f"   🔬 Advanced mathematical concepts enhance predictions")
            else:
                print(f"   💪 Even fallback {total_features} features beat the record!")
                print(f"   🤔 MEGA features had numerical issues but core V1 is solid")
                print(f"   📊 Suggests V1 feature quality is exceptional")
        elif mae < best_mae + 1.0:
            print(f"   🎯 Extremely close! {total_features} features nearly beat record")
            print(f"   💡 Approach shows high potential")
            print(f"   🔧 Minor refinements could push over the top")
        else:
            print(f"   🤔 {total_features} features didn't improve performance")
            if total_features < 50:
                print(f"   ⚠️  Numerical issues forced fallback to basic features")
                print(f"   🔧 Need to fix mega feature numerical stability")
            else:
                print(f"   💭 Possible diminishing returns or feature noise")
        
        print(f"\n🎯 FEATURE ENGINEERING CONCLUSIONS:")
        print(f"   📊 Features tested: {total_features}")
        if total_features > 100:
            print(f"   🔬 Categories: Mathematical, Statistical, Number Theory, Spectral, Chaos, Graph, Temporal")
            print(f"   ⚡ TabPFN performance: {'Excellent' if mae < 60 else 'Good' if mae < 65 else 'Fair'}")
        else:
            print(f"   ⚠️  Had to fall back to core V1 features due to numerical issues")
            print(f"   🔧 MEGA feature engineering needs numerical stability work")
        print(f"   🧠 Next steps: {'Celebrate!' if mae < best_mae else 'Debug numerical issues' if total_features < 50 else 'Feature selection optimization'}")
        
    except ImportError:
        print("❌ TabPFN not available. Please install: pip install tabpfn")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 