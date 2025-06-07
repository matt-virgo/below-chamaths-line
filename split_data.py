#!/usr/bin/env python3

import json
import random
import math

def split_data():
    # Read the public cases data
    with open('public_cases.json', 'r') as f:
        data = json.load(f)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Shuffle the data randomly
    random.shuffle(data)
    
    # Calculate split point (80% for training)
    total_cases = len(data)
    train_size = math.floor(total_cases * 0.8)
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Save training data
    with open('train_cases.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    # Save testing data
    with open('test_cases.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Data split complete!")
    print(f"Total cases: {total_cases}")
    print(f"Training cases: {len(train_data)} ({len(train_data)/total_cases*100:.1f}%)")
    print(f"Testing cases: {len(test_data)} ({len(test_data)/total_cases*100:.1f}%)")
    print(f"Files created: train_cases.json, test_cases.json")

if __name__ == "__main__":
    split_data() 