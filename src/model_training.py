#!/usr/bin/env python3
"""
Model Training and Tracking for Credit Risk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Load data
df = pd.read_csv('../data/processed/model_data_with_proxy.csv')

# Features and target
X = df.drop(columns=['AccountId', 'default_risk', 'risk_category', 'risk_score', 'is_high_risk'], errors='ignore')
y = df['is_high_risk']

# Split
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

mlflow.set_experiment('credit-risk-proxy')

results = []

with mlflow.start_run(run_name="LogisticRegression"):
    params = {'C': [0.01, 0.1, 1, 10]}
    model = GridSearchCV(LogisticRegression(max_iter=500, random_state=random_state), params, cv=3, scoring='f1')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    mlflow.log_params(model.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model.best_estimator_, "model")
    results.append(("LogisticRegression", metrics, model.best_estimator_))

with mlflow.start_run(run_name="RandomForest"):
    params = {'n_estimators': [50, 100], 'max_depth': [3, 5, 10]}
    model = GridSearchCV(RandomForestClassifier(random_state=random_state), params, cv=3, scoring='f1')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    mlflow.log_params(model.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model.best_estimator_, "model")
    results.append(("RandomForest", metrics, model.best_estimator_))

# Select best model by F1
best_model_name, best_metrics, best_model = max(results, key=lambda x: x[1]['f1'])
mlflow.set_experiment('credit-risk-proxy')
with mlflow.start_run(run_name="RegisterBestModel"):
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_metrics(best_metrics)
    mlflow.set_tag("best_model", best_model_name)
    mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/best_model", f"credit-risk-proxy-best")

print(f"Best model: {best_model_name}")
print(f"Metrics: {best_metrics}") 