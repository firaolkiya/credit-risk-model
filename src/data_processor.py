#!/usr/bin/env python3
"""
Main Data Processing Script for Credit Scoring Model
"""

import pandas as pd
import numpy as np
from feature_engineering import load_and_process_data, save_features
from target_generator import create_target_variable, save_targets
import os

def create_model_dataset():
    """Create the final model-ready dataset"""
    
    # Create processed directory if it doesn't exist
    os.makedirs('../data/processed', exist_ok=True)
    
    # Load and process features
    print("Creating features...")
    feature_df, pipeline = load_and_process_data()
    save_features(feature_df, pipeline)
    
    # Create target variables
    print("Creating target variables...")
    df = pd.read_csv('../data/raw/data.csv')
    targets = create_target_variable(df)
    save_targets(targets)
    
    # Merge features and targets
    print("Merging features and targets...")
    model_data = feature_df.merge(targets[['AccountId', 'default_risk', 'risk_category', 'risk_score']], 
                                 on='AccountId', how='inner')
    
    # Save final dataset
    model_data.to_csv('../data/processed/model_data.csv', index=False)
    
    # Create summary
    print(f"\nDataset Summary:")
    print(f"Shape: {model_data.shape}")
    print(f"Features: {len(model_data.columns) - 4}")  # Excluding AccountId and target columns
    print(f"Default rate: {model_data['default_risk'].mean():.3f}")
    print(f"Risk distribution: {model_data['risk_category'].value_counts().to_dict()}")
    
    return model_data

def get_feature_importance_data():
    """Get feature importance analysis data"""
    model_data = pd.read_csv('../data/processed/model_data.csv')
    
    # Separate features and target
    feature_cols = [col for col in model_data.columns if col not in 
                   ['AccountId', 'default_risk', 'risk_category', 'risk_score']]
    
    X = model_data[feature_cols]
    y = model_data['default_risk']
    
    return X, y, feature_cols

if __name__ == "__main__":
    model_data = create_model_dataset()
    print("Data processing completed successfully!") 