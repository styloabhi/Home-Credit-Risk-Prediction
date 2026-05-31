# Home Credit Default Risk Prediction

## Project Overview

This project was developed as an end-to-end Data Science and Business Intelligence solution for the Home Credit Default Risk problem.

The objective is to identify customers who are likely to default on loan repayments using customer demographics, financial information, credit history, and repayment behavior.

The project combines:

* Business Analytics
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning
* Power BI Dashboarding
* Streamlit Deployment
* Pipeline Automation

---

## Business Context

Home Credit aims to provide financial services to individuals with limited or no traditional credit history.

Accurately assessing customer risk is critical for:

* Reducing loan defaults
* Improving portfolio quality
* Increasing profitability
* Supporting responsible lending

---

## Business Problem

Financial institutions face losses due to customer loan defaults.

The primary goals of this project are:

* Predict customer default probability
* Identify key risk drivers
* Generate actionable business insights
* Support decision-making through interactive dashboards

---

## Dataset

### Source

Home Credit Default Risk Dataset

### Main Tables

* Application Train
* Application Test
* Bureau
* Bureau Balance
* Previous Application
* Installments Payments
* Credit Card Balance
* POS Cash Balance

---

## Project Workflow

Business Understanding
↓
Data Exploration
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Machine Learning
↓
Model Evaluation
↓
Business Insights
↓
Dashboard Development
↓
Streamlit Deployment
↓
Pipeline Automation

---

## Exploratory Data Analysis

EDA focused on:

* Customer Demographics
* Income Characteristics
* Credit Exposure
* Repayment Behavior
* Housing Characteristics
* Family Characteristics
* Bureau Credit History
* Delinquency Analysis

### Key Insights

* Lower external credit scores are associated with higher default rates.
* Customers with previous payment delays exhibit significantly higher risk.
* High credit utilization is linked to increased default probability.
* Historical credit performance is a strong predictor of future default behavior.

---

## Feature Engineering

Features were engineered from multiple source tables.

### Bureau Features

* Total Debt
* Active Credit Count
* Overdue Ratios
* Credit History Metrics

### Previous Application Features

* Approval Counts
* Refusal Counts
* Previous Credit Statistics

### Installment Features

* Payment Delays
* Payment Ratios
* Delinquency Metrics

### Credit Card Features

* Credit Utilization
* Outstanding Balances
* Credit Card Delinquency

### POS Features

* POS Delinquency
* Installment Metrics

---

## Machine Learning Models

Models Developed:

* CatBoost
* LightGBM
* XGBoost

### Ensemble Model

Final prediction generated using weighted averaging:

* CatBoost: 50%
* LightGBM: 30%
* XGBoost: 20%

### Objective

Predict customer default probability and classify applicants into risk categories.

---

## Business Dashboards

### Executive Dashboard

Provides:

* Portfolio Overview
* Default Metrics
* Risk Exposure Analysis
* Customer Segmentation

### Risk Analyst Dashboard

Provides:

* Delinquency Analysis
* Credit Utilization Analysis
* Debt Exposure Monitoring
* Risk Segment Performance

### Portfolio Manager Dashboard

Provides:

* Portfolio Exposure
* Loan Distribution
* Installment Burden Analysis
* High-Risk Exposure Monitoring

---

## Streamlit Application

The Streamlit application includes:

### Executive Dashboard

* Portfolio KPIs
* Customer Segmentation
* Risk Analysis

### Risk Analyst Dashboard

* Delinquency Metrics
* Credit Utilization Analysis

### Portfolio Manager Dashboard

* Exposure Analysis
* Portfolio Monitoring

### Customer Default Predictor

* Customer-Level Prediction
* Default Probability Score
* Risk Segmentation
* Credit Recommendation Engine
* Adjustable Decision Threshold

---

## Pipeline Automation

The project includes automated execution using:

* pipeline.py
* pipeline.bat

Pipeline stages:

1. Data Cleaning & EDA
2. Feature Engineering
3. Model Training
4. Output Generation

---

## Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn
* Plotly

### Machine Learning

* Scikit-Learn
* CatBoost
* LightGBM
* XGBoost

### Optimization

* Optuna

### Dashboarding

* Power BI
* Streamlit

---

## Project Structure

Home-Credit-Risk-Prediction

* Data + Problem Statements
* Data Cleaning and EDA Python
* Data Exploration python
* eda_outputs
* Feature engineering and ML
* home_credit_data
* Pipeline and batch
* Power BI Dashboard
* Presentation
* streamlit

---

## Project Deliverables

* Business Analysis
* EDA Reports
* Feature Engineering Framework
* Machine Learning Models
* Ensemble Model
* Power BI Dashboards
* Streamlit Application
* Automated Pipeline
* Project Presentations

---

## Business Recommendations

* Strengthen screening for high-risk customers.
* Utilize external credit scores for decision-making.
* Monitor customers with delinquency history.
* Implement risk-based lending strategies.
* Develop customer risk segmentation frameworks.
* Introduce proactive portfolio monitoring mechanisms.

---

## Author

Abhishek Kumar Pandey

Data Science | Machine Learning | Business Intelligence
