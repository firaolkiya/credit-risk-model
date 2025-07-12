#!/usr/bin/env python3
"""
Target Variable Generator for Credit Scoring Model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_target_variable(df):
    """Create target variable for credit scoring"""
    
    customer_targets = df.groupby('AccountId').agg({
        'FraudResult': ['sum', 'mean'],
        'Value': ['mean', 'max'],
        'Amount': ['mean', 'std']
    }).round(4)
    
    customer_targets.columns = [
        'fraud_count', 'fraud_rate', 'avg_value', 'max_value',
        'avg_amount', 'amount_std'
    ]
    
    customer_targets = customer_targets.reset_index()
    
    # Define risk categories based on fraud patterns and transaction behavior
    customer_targets['risk_score'] = (
        customer_targets['fraud_rate'] * 100 +
        (customer_targets['avg_value'] > customer_targets['avg_value'].quantile(0.9)).astype(int) * 20 +
        (customer_targets['amount_std'] > customer_targets['amount_std'].quantile(0.9)).astype(int) * 10
    )
    
    # Create risk categories
    customer_targets['risk_category'] = pd.cut(
        customer_targets['risk_score'],
        bins=[-np.inf, 10, 30, 50, np.inf],
        labels=['low_risk', 'medium_risk', 'high_risk', 'very_high_risk']
    )
    
    # Binary target for default prediction
    customer_targets['default_risk'] = (
        (customer_targets['fraud_rate'] > 0) |
        (customer_targets['risk_score'] > customer_targets['risk_score'].quantile(0.8))
    ).astype(int)
    
    # Encode risk categories
    le = LabelEncoder()
    customer_targets['risk_category_encoded'] = le.fit_transform(customer_targets['risk_category'])
    
    return customer_targets

def save_targets(target_df):
    """Save target variables"""
    target_df.to_csv('../data/processed/targets.csv', index=False)
    return target_df

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/data.csv')
    targets = create_target_variable(df)
    save_targets(targets)
    print(f"Targets created: {targets.shape}")
    print(f"Risk distribution: {targets['risk_category'].value_counts()}")
    print(f"Default rate: {targets['default_risk'].mean():.3f}") 