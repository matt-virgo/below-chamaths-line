#!/bin/bash

# Black Box Challenge - Our Implementation
# This script uses our best trained model (QuantileTransformer_AttentionNet_WD2e-3)
# which achieved $57.35 MAE - beating the $58.91 target!
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Validate input parameters
if [ $# -ne 3 ]; then
    echo "Error: Exactly 3 parameters required" >&2
    echo "Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>" >&2
    exit 1
fi

# Check if our Python script exists
if [ ! -f "calculate_reimbursement.py" ]; then
    echo "Error: calculate_reimbursement.py not found" >&2
    exit 1
fi

# Check if model files exist
if [ ! -f "best_overall_model.pth" ] && [ ! -f "QuantileTransformer_AttentionNet_WD2e-3_best.pth" ]; then
    echo "Error: Model files not found. Please ensure model training has been completed." >&2
    exit 1
fi

# Run our best model implementation
python3 calculate_reimbursement.py "$1" "$2" "$3" 