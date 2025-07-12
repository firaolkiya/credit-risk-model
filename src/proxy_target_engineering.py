#!/usr/bin/env python3
"""
Proxy Target Variable Engineering for Credit Risk
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(df['TransactionStartTime']).max() + pd.Timedelta(days=1)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
    return rfm

def cluster_rfm(rfm, n_clusters=3, random_state=42):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm, kmeans

def assign_high_risk(rfm):
    cluster_stats = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats.sort_values(['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

def merge_high_risk(main_path, out_path, rfm_high_risk, account_customer_map):
    data = pd.read_csv(main_path)
    # Map AccountId to CustomerId
    data = data.merge(account_customer_map, on='AccountId', how='left')
    data = data.merge(rfm_high_risk, on='CustomerId', how='left')
    data['is_high_risk'] = data['is_high_risk'].fillna(0).astype(int)
    data.drop(columns=['CustomerId'], inplace=True)
    data.to_csv(out_path, index=False)
    return data

if __name__ == "__main__":
    df = pd.read_csv('../data/raw/data.csv')
    rfm = calculate_rfm(df)
    rfm, _ = cluster_rfm(rfm)
    rfm_high_risk = assign_high_risk(rfm)
    # Map AccountId to CustomerId (one-to-one mapping)
    account_customer_map = df[['AccountId', 'CustomerId']].drop_duplicates()
    merge_high_risk('../data/processed/model_data.csv', '../data/processed/model_data_with_proxy.csv', rfm_high_risk, account_customer_map) 