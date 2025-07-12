# Exploratory Data Analysis (EDA) Summary Report
## Credit Scoring Model - Bati Bank Buy Now Pay Later Service

### Executive Summary
This report presents the findings from a comprehensive Exploratory Data Analysis of transaction data for Bati Bank's credit scoring model development. The analysis covers 95,662 transactions from 3,633 unique customers over a 90-day period (November 2018 to February 2019).

---

## 1. Dataset Overview

### Key Statistics
- **Total Transactions**: 95,662
- **Unique Customers**: 3,633
- **Date Range**: November 15, 2018 - February 13, 2019 (90 days)
- **Data Quality**: Excellent - No missing values or duplicates
- **Memory Usage**: 66.48 MB

### Data Structure
The dataset contains 16 fields:
1. TransactionId (unique identifier)
2. BatchId (batch processing identifier)
3. AccountId (customer account identifier)
4. SubscriptionId (subscription identifier)
5. CustomerId (customer identifier)
6. CurrencyCode (all UGX)
7. CountryCode (all 256 - Uganda)
8. ProviderId (6 different providers)
9. ProductId (23 different products)
10. ProductCategory (9 categories)
11. ChannelId (4 channels)
12. Amount (transaction amount)
13. Value (absolute transaction value)
14. TransactionStartTime (timestamp)
15. PricingStrategy (4 strategies)
16. FraudResult (binary fraud indicator)

---

## 2. Data Quality Assessment

### ✅ Excellent Data Quality
- **Missing Values**: 0 (100% complete dataset)
- **Duplicate Rows**: 0 (no duplicates)
- **Data Consistency**: All transactions use UGX currency and CountryCode 256 (Uganda)

### Data Types
- **Numerical Features**: 5 (CountryCode, Amount, Value, PricingStrategy, FraudResult)
- **Categorical Features**: 11 (TransactionId, BatchId, AccountId, etc.)

---

## 3. Key Findings

### 3.1 Customer Behavior Patterns

#### Transaction Distribution
- **Average transactions per customer**: 26.33
- **Median transactions per customer**: 4
- **Range**: 1 to 30,893 transactions per customer
- **Distribution**: Highly skewed - most customers have few transactions, few customers have many

#### Transaction Values
- **Average transaction amount**: 6,717.85 UGX
- **Median transaction amount**: 1,000 UGX
- **Range**: -1,000,000 to 9,880,000 UGX
- **Distribution**: Right-skewed with many small transactions and few large ones

### 3.2 Fraud Analysis

#### Overall Fraud Rate
- **Fraud Rate**: 0.20% (193 fraudulent transactions out of 95,662)
- **Fraud Amount**: Average 1,535,272 UGX per fraudulent transaction
- **Legitimate Amount**: Average 3,628 UGX per legitimate transaction

#### Fraud by Product Category
| Product Category | Total Transactions | Fraud Count | Fraud Rate |
|------------------|-------------------|-------------|------------|
| transport | 25 | 2 | 8.00% |
| utility_bill | 1,920 | 12 | 0.63% |
| financial_services | 45,405 | 161 | 0.35% |
| airtime | 45,027 | 18 | 0.04% |
| data_bundles | 1,613 | 0 | 0.00% |
| movies | 175 | 0 | 0.00% |
| other | 2 | 0 | 0.00% |
| ticket | 216 | 0 | 0.00% |
| tv | 1,279 | 0 | 0.00% |

### 3.3 Correlation Analysis

#### Strong Correlations
1. **Amount vs Value**: 0.99 (almost perfect correlation)
2. **Value vs FraudResult**: 0.57 (moderate positive correlation)
3. **Amount vs FraudResult**: 0.56 (moderate positive correlation)

#### Key Insight
Higher transaction values are associated with higher fraud risk, making transaction value a strong predictor for fraud detection.

### 3.4 Product Category Analysis

#### Most Popular Categories
1. **financial_services**: 45,405 transactions (47.5%)
2. **airtime**: 45,027 transactions (47.1%)
3. **utility_bill**: 1,920 transactions (2.0%)
4. **data_bundles**: 1,613 transactions (1.7%)
5. **tv**: 1,279 transactions (1.3%)

#### Risk Assessment by Category
- **High Risk**: transport (8% fraud rate)
- **Medium Risk**: utility_bill (0.63% fraud rate), financial_services (0.35% fraud rate)
- **Low Risk**: airtime (0.04% fraud rate)
- **No Risk**: data_bundles, movies, other, ticket, tv (0% fraud rate)

---

## 4. Top 5 Key Insights

### 1. **Customer Concentration Risk**
- One customer (AccountId_4841) has 30,893 transactions (32.3% of all transactions)
- Top 5 customers account for 43.8% of all transactions
- **Implication**: Need to handle customer concentration in risk modeling

### 2. **Transaction Value as Fraud Indicator**
- Fraudulent transactions average 1.5M UGX vs 3.6K UGX for legitimate transactions
- 423x higher average value for fraudulent transactions
- **Implication**: Transaction value is a critical fraud predictor

### 3. **Product Category Risk Stratification**
- Clear risk hierarchy: transport > utility_bill > financial_services > airtime
- Some categories (data_bundles, tv, movies) have zero fraud
- **Implication**: Product category should be a key feature in credit scoring

### 4. **Temporal Patterns**
- 90-day transaction period shows consistent daily patterns
- Multiple transactions per day suggest active user base
- **Implication**: Time-based features will be valuable for risk assessment

### 5. **Provider Distribution**
- ProviderId_4 (38,189 transactions) and ProviderId_6 (34,186 transactions) dominate
- **Implication**: Provider-specific risk patterns may exist

---

## 5. Recommendations for Feature Engineering

### 5.1 RFM Features (Recency, Frequency, Monetary)
1. **Recency**: Days since last transaction for each customer
2. **Frequency**: Number of transactions per customer (already calculated)
3. **Monetary**: Total amount spent, average transaction value, transaction value volatility

### 5.2 Risk Indicators
1. **Fraud History**: Previous fraud incidents per customer
2. **High-Value Transaction Ratio**: Percentage of transactions above certain thresholds
3. **Product Risk Score**: Weighted average of product category fraud rates
4. **Provider Risk Score**: Provider-specific fraud rates

### 5.3 Behavioral Features
1. **Transaction Patterns**: Time of day, day of week preferences
2. **Category Preferences**: Most purchased product categories
3. **Channel Usage**: Preferred payment channels
4. **Transaction Velocity**: Transactions per day/week

### 5.4 Temporal Features
1. **Seasonality**: Monthly and weekly patterns
2. **Growth Trends**: Increasing/decreasing transaction frequency
3. **Gap Analysis**: Time between transactions

---

## 6. Next Steps for Model Development

### 6.1 Proxy Variable Definition
Based on the analysis, recommend defining credit risk using:
- **FraudResult** as primary indicator
- **High-value transaction patterns** as secondary indicator
- **Product category risk scores** as tertiary indicator

### 6.2 Feature Selection Priority
1. **High Priority**: Transaction Value, Product Category, Fraud History
2. **Medium Priority**: Transaction Frequency, Provider, Channel
3. **Low Priority**: Time-based features, Customer demographics

### 6.3 Model Considerations
- **Class Imbalance**: Only 0.2% fraud rate requires special handling
- **Customer Concentration**: Need to account for heavy users
- **Feature Engineering**: Focus on RFM and risk-based features
- **Validation Strategy**: Time-based split recommended

---

## 7. Conclusion

The dataset provides a solid foundation for credit scoring model development with:
- **Excellent data quality** (no missing values or duplicates)
- **Clear fraud patterns** (value-based indicators)
- **Rich behavioral data** (RFM patterns, product preferences)
- **Temporal consistency** (90-day transaction history)

The analysis reveals that transaction value and product category are the strongest predictors of fraud risk, making them ideal candidates for the credit scoring model's feature set.

**Recommendation**: Proceed with model development focusing on value-based risk indicators and product category risk stratification. 