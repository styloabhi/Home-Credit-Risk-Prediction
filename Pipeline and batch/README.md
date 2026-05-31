# Home Credit Default Risk Prediction

## Business Context

Home Credit aims to expand financial inclusion by providing loans to individuals with limited or no credit history.

The challenge is to identify customers who are likely to default on their loans while ensuring that creditworthy applicants are not rejected.

This project combines business analytics, data visualization, machine learning, and dashboarding to support data-driven lending decisions.

---

## Business Problem

Financial institutions face significant losses due to loan defaults.

The objective of this project is to:

* Predict customer default risk
* Identify key drivers of default
* Generate actionable business insights
* Support decision-making through dashboards and visual analytics

---

## Project Objective

The primary objective of this project is to develop a data-driven solution that helps Home Credit identify customers at risk of loan default while providing actionable business insights through data analysis, dashboards, and machine learning models.

---

## Key Deliverables

* Data Cleaning & Preparation
* Exploratory Data Analysis (EDA)
* Business Insights & Recommendations
* Power BI Dashboard
* Streamlit Dashboard
* Machine Learning Models
* Automated Pipeline
* Final Project Presentation (PPT)

---

## Dataset Description

The project uses the Home Credit Default Risk dataset.

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

## Data Dictionary

The dataset contains customer information related to:

* Demographics
* Income
* Employment
* Credit History
* Loan Applications
* Installment Payments
* Credit Card Usage

---

## Project Workflow

```text
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
Business Insights
        ↓
Dashboard Development
        ↓
Presentation & Recommendations
```

---

## Exploratory Data Analysis

EDA was performed to:

* Understand customer characteristics
* Analyze default patterns
* Identify high-risk customer segments
* Discover relationships between variables
* Generate business insights

### Areas Explored

* Income Analysis
* Employment Analysis
* Credit Analysis
* Family Characteristics
* Housing Characteristics
* Loan Repayment Behavior
* Bureau Credit History

---

## Business Insights

Key insights were extracted from EDA and dashboard analysis.

Examples include:

* High credit burden customers exhibit higher default rates.
* Certain income categories demonstrate elevated risk.
* Historical repayment behavior is a strong predictor of default.
* Previous loan performance significantly impacts future default probability.

---

## Business Recommendations

### Risk Management

* Strengthen screening for high-risk applicants.
* Introduce risk-based lending strategies.

### Customer Segmentation

* Create customer risk tiers.
* Develop differentiated lending policies.

### Credit Monitoring

* Monitor customers with adverse credit history.
* Implement early warning systems.

---

## Machine Learning Solution

### Models Developed

* LightGBM
* CatBoost
* XGBoost
* Ensemble Model

### Objective

Predict the probability of customer loan default using demographic, financial, and credit history information.

---

## Dashboards

### Power BI Dashboard

Includes:

* Portfolio Overview
* Default Analysis
* Customer Segmentation
* Risk Distribution
* Credit Profile Analysis

### Streamlit Dashboard

Provides:

* Interactive exploration
* Customer-level insights
* Risk analysis
* Model predictions and visualizations

---

## Presentation Deliverables

### EDA Presentation

* Dataset Overview
* Exploratory Analysis
* Key Insights
* Business Findings

### Business Presentation

* Problem Statement
* Findings
* Recommendations
* Strategic Impact

---

## Automation Pipeline

The project includes an automated pipeline that executes:

1. Data Cleaning & EDA Notebook
2. Feature Engineering & ML Notebook

### Pipeline Components

* pipeline.py
* pipeline.bat

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Pipeline

### Option 1

```bash
python pipeline.py
```

### Option 2

Double-click:

```text
pipeline.bat
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* LightGBM
* XGBoost
* CatBoost
* Optuna
* Power BI
* Streamlit
* Jupyter Notebook

---

## Project Outputs

### Analytics Deliverables

* Data Cleaning Report
* Exploratory Data Analysis
* Business Insights
* Business Recommendations

### Dashboard Deliverables

* Power BI Dashboard
* Streamlit Dashboard

### Machine Learning Deliverables

* LightGBM Model
* CatBoost Model
* XGBoost Model
* Ensemble Model

### Presentation Deliverables

* EDA Presentation
* Business Presentation
* Final Project PPT

---

## Future Enhancements

* Model Deployment
* Real-Time Prediction System
* Automated Monitoring Dashboard
* MLOps Integration
* Cloud Deployment

---

## Author

**Abhishek Kumar Pandey**

Data Science Capstone Project – Home Credit Default Risk Prediction