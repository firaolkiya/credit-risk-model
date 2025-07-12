#!/usr/bin/env python3
"""
Exploratory Data Analysis (EDA) for Credit Scoring Model
Bati Bank - Buy Now Pay Later Service
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

def load_data():
    """Load the dataset and perform initial exploration"""
    print("Loading dataset...")
    df = pd.read_csv('../data/raw/data.csv')
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def basic_overview(df):
    """Perform basic dataset overview"""
    print("\n" + "="*50)
    print("DATASET OVERVIEW")
    print("="*50)
    
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    
    print("\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nFirst 5 rows:")
    print(df.head())

def data_quality_assessment(df):
    """Assess data quality including missing values and duplicates"""
    print("\n" + "="*50)
    print("DATA QUALITY ASSESSMENT")
    print("="*50)
    
    # Missing values analysis
    print("Missing Values Analysis:")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percentage
    })
    
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    if missing_df.empty:
        print("✅ No missing values found in the dataset!")
    else:
        print(missing_df)
    
    # Duplicate analysis
    print(f"\nDuplicate Analysis:")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    print(f"Duplicate percentage: {(duplicates/len(df))*100:.2f}%")

def summary_statistics(df):
    """Generate summary statistics for numerical and categorical features"""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    # Numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Numerical columns: {list(numerical_cols)}")
    
    print("\nSummary Statistics for Numerical Features:")
    print(df[numerical_cols].describe())
    
    # Categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"\nCategorical columns: {list(categorical_cols)}")
    
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
        print(f"  Top 5 values:")
        print(df[col].value_counts().head())

def distribution_analysis(df):
    """Analyze distributions of numerical and categorical features"""
    print("\n" + "="*50)
    print("DISTRIBUTION ANALYSIS")
    print("="*50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create distribution plots for numerical features
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols[:6]):
        axes[i].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../notebooks/numerical_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Box plots for outlier detection
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(numerical_cols[:6]):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Box Plot of {col}')
        axes[i].set_ylabel(col)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../notebooks/outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def correlation_analysis(df):
    """Perform correlation analysis on numerical features"""
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix of Numerical Features')
    plt.tight_layout()
    plt.savefig('../notebooks/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display top correlations
    print("Top Correlations:")
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            correlations.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))
    
    correlations_df = pd.DataFrame(correlations, columns=['Feature1', 'Feature2', 'Correlation'])
    correlations_df = correlations_df.sort_values('Correlation', key=abs, ascending=False)
    print(correlations_df.head(10))

def time_series_analysis(df):
    """Analyze time-based patterns in the data"""
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS")
    print("="*50)
    
    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    
    # Extract time-based features
    df['Year'] = df['TransactionStartTime'].dt.year
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Day'] = df['TransactionStartTime'].dt.day
    df['DayOfWeek'] = df['TransactionStartTime'].dt.dayofweek
    df['Hour'] = df['TransactionStartTime'].dt.hour
    
    print(f"Date range: {df['TransactionStartTime'].min()} to {df['TransactionStartTime'].max()}")
    print(f"Total days: {(df['TransactionStartTime'].max() - df['TransactionStartTime'].min()).days}")
    
    # Transaction volume over time
    daily_transactions = df.groupby(df['TransactionStartTime'].dt.date).size()
    
    plt.figure(figsize=(15, 6))
    daily_transactions.plot(kind='line')
    plt.title('Daily Transaction Volume')
    plt.xlabel('Date')
    plt.ylabel('Number of Transactions')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../notebooks/daily_transactions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def customer_behavior_analysis(df):
    """Analyze customer behavior patterns"""
    print("\n" + "="*50)
    print("CUSTOMER BEHAVIOR ANALYSIS")
    print("="*50)
    
    # Customer-level analysis
    customer_stats = df.groupby('AccountId').agg({
        'TransactionId': 'count',
        'Amount': ['sum', 'mean', 'std'],
        'Value': ['sum', 'mean'],
        'TransactionStartTime': ['min', 'max']
    }).round(2)
    
    customer_stats.columns = ['Transaction_Count', 'Total_Amount', 'Avg_Amount', 'Std_Amount', 
                             'Total_Value', 'Avg_Value', 'First_Transaction', 'Last_Transaction']
    
    print("Customer Statistics Summary:")
    print(customer_stats.describe())
    
    # Distribution of transaction counts per customer
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(customer_stats['Transaction_Count'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Transactions per Customer')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Customers')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(customer_stats['Total_Amount'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Total Amount per Customer')
    plt.xlabel('Total Amount')
    plt.ylabel('Number of Customers')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../notebooks/customer_behavior.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return customer_stats

def fraud_analysis(df):
    """Analyze fraud patterns in the data"""
    print("\n" + "="*50)
    print("FRAUD ANALYSIS")
    print("="*50)
    
    # Fraud statistics
    fraud_stats = df.groupby('FraudResult').agg({
        'TransactionId': 'count',
        'Amount': ['sum', 'mean'],
        'Value': ['sum', 'mean']
    }).round(2)
    
    fraud_stats.columns = ['Transaction_Count', 'Total_Amount', 'Avg_Amount', 
                          'Total_Value', 'Avg_Value']
    
    print("Fraud Statistics:")
    print(fraud_stats)
    
    # Fraud rate
    fraud_rate = (df['FraudResult'].sum() / len(df)) * 100
    print(f"\nFraud rate: {fraud_rate:.2f}%")
    
    # Fraud by category
    fraud_by_category = df.groupby('ProductCategory')['FraudResult'].agg(['count', 'sum', 'mean'])
    fraud_by_category.columns = ['Total_Transactions', 'Fraud_Count', 'Fraud_Rate']
    fraud_by_category = fraud_by_category.sort_values('Fraud_Rate', ascending=False)
    
    print("\nFraud by Product Category (Top 10):")
    print(fraud_by_category.head(10))
    
    return fraud_rate, fraud_by_category

def generate_insights(df, customer_stats, fraud_rate):
    """Generate key insights from the analysis"""
    print("\n" + "="*50)
    print("KEY INSIGHTS SUMMARY")
    print("="*50)
    
    print("\n1. DATASET OVERVIEW:")
    print(f"   - Total transactions: {len(df):,}")
    print(f"   - Unique customers: {df['AccountId'].nunique():,}")
    print(f"   - Date range: {df['TransactionStartTime'].min().strftime('%Y-%m-%d')} to {df['TransactionStartTime'].max().strftime('%Y-%m-%d')}")
    
    print("\n2. DATA QUALITY:")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    print("\n3. FRAUD PATTERNS:")
    print(f"   - Overall fraud rate: {fraud_rate:.2f}%")
    print(f"   - Fraud transactions: {df['FraudResult'].sum():,}")
    
    print("\n4. CUSTOMER BEHAVIOR:")
    print(f"   - Average transactions per customer: {customer_stats['Transaction_Count'].mean():.2f}")
    print(f"   - Average amount per transaction: {df['Amount'].mean():.2f}")
    
    # Top 3 insights
    print("\n5. TOP 3 INSIGHTS:")
    print("   a) Customer Engagement: Most customers have moderate transaction frequency")
    print("   b) Fraud Risk: Low overall fraud rate but varies by product category")
    print("   c) Transaction Patterns: Clear temporal patterns in transaction volume")

def main():
    """Main function to run the complete EDA"""
    print("Starting Exploratory Data Analysis for Credit Scoring Model")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Perform analysis
    basic_overview(df)
    data_quality_assessment(df)
    summary_statistics(df)
    distribution_analysis(df)
    correlation_analysis(df)
    
    # Time series analysis
    df = time_series_analysis(df)
    
    # Customer behavior analysis
    customer_stats = customer_behavior_analysis(df)
    
    # Fraud analysis
    fraud_rate, fraud_by_category = fraud_analysis(df)
    
    # Generate insights
    generate_insights(df, customer_stats, fraud_rate)
    
    print("\n" + "="*60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return df, customer_stats, fraud_rate, fraud_by_category

if __name__ == "__main__":
    main() 