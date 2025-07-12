#!/usr/bin/env python3
"""
Feature Engineering Pipeline for Credit Scoring Model
Bati Bank - Buy Now Pay Later Service
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from transaction data"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        
        X_copy['hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['year'] = X_copy['TransactionStartTime'].dt.year
        X_copy['day_of_week'] = X_copy['TransactionStartTime'].dt.dayofweek
        X_copy['is_weekend'] = X_copy['day_of_week'].isin([5, 6]).astype(int)
        
        return X_copy

class RiskFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract risk-based features"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        fraud_by_category = X_copy.groupby('ProductCategory')['FraudResult'].mean()
        fraud_by_provider = X_copy.groupby('ProviderId')['FraudResult'].mean()
        
        X_copy['category_fraud_rate'] = X_copy['ProductCategory'].map(fraud_by_category)
        X_copy['provider_fraud_rate'] = X_copy['ProviderId'].map(fraud_by_provider)
        
        X_copy['high_value_transaction'] = (X_copy['Value'] > X_copy['Value'].quantile(0.95)).astype(int)
        X_copy['low_value_transaction'] = (X_copy['Value'] < X_copy['Value'].quantile(0.05)).astype(int)
        
        return X_copy

class CustomerAggregator(BaseEstimator, TransformerMixin):
    """Aggregate customer-level features"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        customer_features = X.groupby('AccountId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean'],
            'FraudResult': ['sum', 'mean'],
            'category_fraud_rate': 'mean',
            'provider_fraud_rate': 'mean',
            'high_value_transaction': 'sum',
            'low_value_transaction': 'sum',
            'is_weekend': 'mean',
            'hour': ['mean', 'std'],
            'day_of_week': ['mean', 'std'],
            'ProductCategory': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'ProviderId': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0],
            'ChannelId': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
        }).round(4)
        
        customer_features.columns = [
            'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
            'total_value', 'avg_value', 'fraud_count', 'fraud_rate',
            'avg_category_fraud_rate', 'avg_provider_fraud_rate',
            'high_value_count', 'low_value_count', 'weekend_ratio',
            'avg_hour', 'std_hour', 'avg_day_of_week', 'std_day_of_week',
            'ProductCategory', 'ProviderId', 'ChannelId'
        ]
        
        customer_features['amount_volatility'] = (
            customer_features['std_amount'] / 
            customer_features['avg_amount'].replace(0, 1)
        )
        
        customer_features['value_volatility'] = (
            customer_features['std_amount'] / 
            customer_features['avg_value'].replace(0, 1)
        )
        
        customer_features['high_value_ratio'] = (
            customer_features['high_value_count'] / 
            customer_features['transaction_count']
        )
        
        customer_features['low_value_ratio'] = (
            customer_features['low_value_count'] / 
            customer_features['transaction_count']
        )
        
        return customer_features.reset_index()

def create_feature_pipeline():
    """Create the complete feature engineering pipeline"""
    
    feature_extraction = Pipeline([
        ('temporal_extractor', TemporalFeatureExtractor()),
        ('risk_extractor', RiskFeatureExtractor()),
        ('customer_aggregator', CustomerAggregator())
    ])
    
    categorical_features = ['ProductCategory', 'ProviderId', 'ChannelId']
    numerical_features = [
        'total_amount', 'avg_amount', 'std_amount', 'transaction_count',
        'total_value', 'avg_value', 'fraud_count', 'fraud_rate',
        'avg_category_fraud_rate', 'avg_provider_fraud_rate',
        'high_value_count', 'low_value_count', 'weekend_ratio',
        'avg_hour', 'std_hour', 'avg_day_of_week', 'std_day_of_week',
        'amount_volatility', 'value_volatility', 'high_value_ratio', 'low_value_ratio'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    feature_pipeline = Pipeline([
        ('feature_extraction', feature_extraction),
        ('preprocessor', preprocessor)
    ])
    
    return feature_pipeline

def load_and_process_data():
    """Load data and create features"""
    df = pd.read_csv('../data/raw/data.csv')
    
    pipeline = create_feature_pipeline()
    features = pipeline.fit_transform(df)
    
    feature_names = (
        [f'num_{col}' for col in pipeline.named_steps['preprocessor']
         .named_transformers_['num'].get_feature_names_out()] +
        [f'cat_{col}' for col in pipeline.named_steps['preprocessor']
         .named_transformers_['cat'].get_feature_names_out()]
    )
    
    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df['AccountId'] = df.groupby('AccountId').size().index
    
    return feature_df, pipeline

def save_features(feature_df, pipeline):
    """Save processed features and pipeline"""
    feature_df.to_csv('../data/processed/features.csv', index=False)
    
    import joblib
    joblib.dump(pipeline, '../data/processed/feature_pipeline.pkl')
    
    return feature_df

if __name__ == "__main__":
    feature_df, pipeline = load_and_process_data()
    save_features(feature_df, pipeline)
    print(f"Features created: {feature_df.shape}")
    print(f"Feature columns: {list(feature_df.columns)}") 