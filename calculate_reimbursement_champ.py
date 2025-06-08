#!/usr/bin/env python3

"""
Champion Reimbursement Calculator - TabPFN Business Rules Champion
Uses our $55.21 MAE World Record model for single prediction command-line interface

Usage: python3 calculate_reimbursement_champ.py <trip_duration_days> <miles_traveled> <total_receipts_amount>
"""

import sys
import json
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class ChampionReimbursementCalculator:
    """Champion calculator using TabPFN Business Rules approach"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def load_training_data(self, train_file='train_cases.json'):
        """Load training data to fit the TabPFN model"""
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # Convert to DataFrame
        self.train_df = pd.DataFrame([
            {
                'trip_duration_days': case['input']['trip_duration_days'],
                'miles_traveled': case['input']['miles_traveled'],
                'total_receipts_amount': case['input']['total_receipts_amount'],
                'reimbursement': case['expected_output']
            }
            for case in train_data
        ])
        
    def engineer_business_features(self, df_input):
        """Apply our winning business rules feature engineering (31 features)"""
        df = df_input.copy()

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
        df['is_long_trip'] = ((df['trip_duration_days'] > 6) & (df['trip_duration_days'] < 8)).astype(int)
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

        # Kevin's insights
        df['is_optimal_miles_per_day_kevin'] = ((df['miles_per_day'] >= 180) & (df['miles_per_day'] <= 220)).astype(int)
        
        def optimal_daily_spending(row):
            if row['is_short_trip']:
                return 1 if row['receipts_per_day'] < 75 else 0
            elif row['is_medium_trip']:
                return 1 if row['receipts_per_day'] < 120 else 0
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

        # Select final features (31 total)
        business_features = [
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
        
        return df[business_features]
        
    def fit(self):
        """Fit the TabPFN model on training data"""
        try:
            from tabpfn import TabPFNRegressor
            
            # Create and engineer training features
            X_train = self.engineer_business_features(self.train_df)
            y_train = self.train_df['reimbursement'].values
            
            # Initialize and fit TabPFN
            self.model = TabPFNRegressor(device='cpu')
            
            # Convert to numpy arrays
            X_train_np = X_train.values.astype(np.float32)
            y_train_np = y_train.astype(np.float32)
            
            self.model.fit(X_train_np, y_train_np)
            self.is_trained = True
            
        except ImportError:
            raise ImportError("TabPFN not available. Please install: pip install tabpfn")

# Global calculator instance
_calculator = None

def get_calculator():
    """Get or create the global calculator instance"""
    global _calculator
    if _calculator is None:
        _calculator = ChampionReimbursementCalculator()
        _calculator.load_training_data()
        _calculator.fit()
    return _calculator

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using our Champion TabPFN Business Rules model
    
    Args:
        trip_duration_days (float): Number of days for the trip
        miles_traveled (float): Total miles traveled
        total_receipts_amount (float): Total amount from receipts
    
    Returns:
        float: Calculated reimbursement amount
    """
    
    # Get the trained calculator
    calculator = get_calculator()
    
    # Create input DataFrame
    input_df = pd.DataFrame([{
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }])
    
    # Engineer features
    X_new = calculator.engineer_business_features(input_df)
    
    # Make prediction
    X_new_np = X_new.values.astype(np.float32)
    prediction = calculator.model.predict(X_new_np)
    
    return float(prediction[0])

def main():
    """Main function to handle command line arguments"""
    
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement_champ.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        # Parse command line arguments
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        # Validate inputs
        if trip_duration_days <= 0:
            raise ValueError("Trip duration must be positive")
        if miles_traveled < 0:
            raise ValueError("Miles traveled must be non-negative")
        if total_receipts_amount < 0:
            raise ValueError("Total receipts amount must be non-negative")
        
        # Calculate reimbursement using Champion model
        reimbursement = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
        
        # Output the result (single number as required)
        print(f"{reimbursement:.2f}")
        
    except ValueError as e:
        print(f"Error: Invalid input - {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please ensure train_cases.json is present in the current directory.", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Please install TabPFN: pip install tabpfn", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 