# Home Credit Pipeline Automation

## Overview

This pipeline is part of the Home Credit Default Risk Prediction project.

The main project focuses on predicting customer loan default risk using machine learning, business intelligence dashboards, and automated data processing.

This folder contains the automation pipeline used to execute the project workflow efficiently.

---

## Purpose

The objective of the pipeline is to automate repetitive tasks involved in the machine learning workflow.

Instead of manually running multiple notebooks and scripts, the pipeline executes the required steps in sequence and generates the final outputs.

---

## Pipeline Workflow

Raw Data
↓
Data Cleaning
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Output Generation

---

## Components

### pipeline.py

Main Python script responsible for:

* Loading raw datasets
* Executing preprocessing steps
* Running feature engineering
* Training machine learning models
* Saving outputs and artifacts

### pipeline.bat

Windows batch file used to execute the pipeline with a single click.

---

## Input Data

The pipeline uses Home Credit datasets including:

* Application Data
* Bureau Data
* Previous Application Data
* Installment Payment Data
* Credit Card Balance Data
* POS Cash Balance Data

---

## Generated Outputs

The pipeline produces:

* Cleaned datasets
* Engineered features
* Trained machine learning models
* Evaluation metrics
* Model artifacts for deployment

---

## Machine Learning Models

The project pipeline supports:

* CatBoost
* LightGBM
* XGBoost

The final prediction system uses an ensemble approach combining all three models.

---

## How to Run

### Option 1: Python

```bash
python pipeline.py
```

### Option 2: Batch File

```bash
pipeline.bat
```

---

## Project Context

This pipeline is a component of the larger Home Credit Default Risk Prediction project, which includes:

* Exploratory Data Analysis
* Feature Engineering
* Machine Learning
* Power BI Dashboards
* Streamlit Dashboard Application
* Business Insights and Recommendations

---

## Author

Abhishek Kumar Pandey
