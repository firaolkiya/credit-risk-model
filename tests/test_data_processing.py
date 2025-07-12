import pandas as pd
from src.proxy_target_engineering import calculate_rfm

def test_rfm_shape():
    data = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B'],
        'TransactionStartTime': ['2024-01-01', '2024-01-10', '2024-01-05'],
        'TransactionId': [1, 2, 3],
        'Value': [100, 200, 300]
    })
    rfm = calculate_rfm(data, snapshot_date=pd.Timestamp('2024-01-15'))
    assert rfm.shape[0] == 2
    assert set(rfm.columns) == {'CustomerId', 'Recency', 'Frequency', 'Monetary'}

def test_rfm_values():
    data = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B'],
        'TransactionStartTime': ['2024-01-01', '2024-01-10', '2024-01-05'],
        'TransactionId': [1, 2, 3],
        'Value': [100, 200, 300]
    })
    rfm = calculate_rfm(data, snapshot_date=pd.Timestamp('2024-01-15'))
    a = rfm[rfm['CustomerId'] == 'A'].iloc[0]
    b = rfm[rfm['CustomerId'] == 'B'].iloc[0]
    assert a['Recency'] == 5  # 2024-01-15 - 2024-01-10
    assert a['Frequency'] == 2
    assert a['Monetary'] == 300
    assert b['Recency'] == 10  # 2024-01-15 - 2024-01-05
    assert b['Frequency'] == 1
    assert b['Monetary'] == 300 