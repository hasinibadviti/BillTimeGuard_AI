# BillTimeGuard AI – AI-Based Legal Billing Anomaly Detector

An AI-powered legal billing anomaly detection system that identifies potentially unusual patterns in legal time-tracking and invoicing records using an Autoencoder-based unsupervised machine learning approach.

## Project Overview

BillTimeGuard AI is designed to help identify unusual billing patterns in legal time entries. Traditional manual billing reviews can be time-consuming, especially when dealing with large numbers of billing records.

This project uses an Autoencoder neural network to learn patterns from legal billing data and identify records that differ significantly from the learned behaviour.

The system calculates reconstruction errors for billing records and compares them against a predefined anomaly threshold. Records exceeding the threshold are flagged as potential anomalies for further investigation.

The project also provides a web-based dashboard, a FastAPI backend, billing analysis and PDF report generation.

## Problem Statement

Law firms handle large volumes of time-tracking and invoicing records. Manual examination of every billing entry can be inefficient and may make unusual billing patterns difficult to identify.

BillTimeGuard AI addresses this problem by using unsupervised machine learning to identify potentially unusual billing behaviour and support more efficient billing review.

## Objectives

- Prepare and analyse legal billing data.
- Perform data preprocessing and feature engineering.
- Develop an Autoencoder-based anomaly detection model.
- Calculate reconstruction errors and establish an anomaly threshold.
- Identify potentially unusual billing records.
- Integrate the trained model with a FastAPI backend.
- Develop a web-based billing analysis dashboard.
- Generate downloadable PDF analysis reports.
- Evaluate model behaviour using controlled anomaly scenarios.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | TensorFlow, Keras |
| Data Processing | Pandas, NumPy |
| Data Preprocessing | Scikit-learn |
| Data Visualization | Matplotlib |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| PDF Generation | ReportLab |
| Model Storage | Keras, Joblib |
| Development Environment | VS Code |

## Dataset

**Dataset:** LexPipe – Time Entries and Phases

**Source:** https://huggingface.co/datasets/lexpipe/time-entries-and-phases

The project uses legal time-entry records containing information about billing duration, hourly rates, professional positions and other billing-related characteristics.

### Dataset Preparation

- Original dataset: 12,417 records.
- Processed dataset: 4,166 records.
- Test dataset: 602 records.
- Missing and unreliable records were handled during preprocessing.
- Billing amount was calculated using billing duration multiplied by hourly rate.
- Numerical features were standardized.
- Categorical features were encoded using OneHotEncoder.

### Features Used

- Entry Duration Hours
- Hourly Billing Rate
- Day of Week
- Month
- Partner Status
- Professional Position

## System Architecture

The project follows the workflow below:

1. **Dataset Preparation:** Legal billing records are cleaned and prepared.
2. **Feature Engineering:** Relevant numerical and categorical billing features are extracted.
3. **Model Training:** An Autoencoder is trained to reconstruct billing records.
4. **Anomaly Detection:** Reconstruction errors are calculated for input records.
5. **Threshold Comparison:** Anomaly scores are compared with the selected threshold.
6. **API Integration:** FastAPI connects the trained model with the application.
7. **Dashboard:** Users enter billing details and view analysis results.
8. **PDF Reporting:** Analysis results can be downloaded as PDF reports.

## Autoencoder Model

The project uses an Autoencoder neural network for unsupervised anomaly detection.

The architecture consists of:

- Input Layer: 16 features
- Encoder Layer 1: 128 neurons with ReLU activation
- Encoder Layer 2: 64 neurons with ReLU activation
- Bottleneck Layer: 32 neurons with ReLU activation
- Decoder Layer 1: 64 neurons with ReLU activation
- Decoder Layer 2: 128 neurons with ReLU activation
- Output Layer: 16 neurons

**Training Configuration:**

- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Learning Rate: 0.001
- Training Epochs: Up to 50
- Batch Size: 256
- Early Stopping and learning-rate reduction callbacks

### Anomaly Detection Mechanism

The model reconstructs each billing record and calculates the Mean Squared Error between the original input and reconstructed output.

The selected anomaly threshold is:

`0.0028734242`

Classification:

- Reconstruction Error ≤ Threshold: Normal Range
- Reconstruction Error > Threshold: Potential Anomaly

## Features

### 1. AI-Based Anomaly Detection
Uses a trained Autoencoder to calculate anomaly scores and identify potentially unusual billing records.

### 2. Billing Analysis
Examines billing duration, hourly rates, billed amounts and other billing characteristics.

### 3. Interactive Dashboard
Provides a web interface for entering billing information and viewing anomaly detection results.

### 4. FastAPI Integration
Connects the trained machine learning model to the frontend through API endpoints.

### 5. Review Indicators
Displays basic rule-based billing indicators to support manual examination.

### 6. Analysis History
Maintains browser-based analysis history for reviewing previous results.

### 7. CSV Export
Allows users to export analysis history.

### 8. PDF Report Generation
Generates downloadable PDF reports containing billing information, anomaly scores, thresholds and analysis results.

## Model Evaluation Results

The trained model was evaluated using 602 test records.

| Evaluation Metric | Result |
|---|---:|
| Total Test Records | 602 |
| Normal-Range Records | 573 |
| Potential Anomaly Records | 29 |
| Anomaly Threshold | 0.0028734242 |

### Controlled Anomaly Scenario Testing

Additional experiments were conducted by artificially modifying hourly billing rates and billing durations.

| Scenario | Flagged Records | Detection Rate |
|---|---:|---:|
| Original | 29 | 4.82% |
| Rate Only (3×) | 581 | 96.51% |
| Duration Only (5×) | 112 | 18.60% |
| Combined | 583 | 96.84% |

These experiments demonstrate how the model responds to selected billing modifications. The results represent controlled synthetic anomaly testing, not verified real-world fraud detection accuracy.

## Project Structure

```text
BillTimeGuard_AI/
│
├── api/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── dataset/
│   ├── lexpipe/
│   ├── processed/
│   └── visualizations/
│
├── models/
│   ├── autoencoder.py
│   ├── train_autoencoder.py
│   ├── evaluate_autoencoder.py
│   ├── test_simulated_anomalies.py
│   ├── evaluate_anomaly_scenarios.py
│   └── saved/
│
├── reports/
│   ├── legal_anomaly_results.csv
│   ├── legal_anomaly_threshold.csv
│   ├── flagged_legal_entries.csv
│   ├── billing_characteristics_comparison.csv
│   ├── anomaly_scenario_comparison.csv
│   └── anomaly_scenario_comparison.png
│
├── notebooks/
│   └── notebooks/
│       └── preprocess_legal_data.py
│
├── requirements.txt
└── README.md
```

*Note: The structure above highlights the main project files. Generated datasets, model files and reports may contain additional files and subdirectories.*

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project folder:

```bash
cd BillTimeGuard_AI
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the virtual environment:

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows:**

```bash
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

If the requirements file is not included, install the required packages manually:

```bash
pip install numpy pandas scikit-learn tensorflow fastapi uvicorn joblib matplotlib reportlab
```

## Running the Application

### Step 1: Start the Backend

From the main project directory, activate the virtual environment and execute:

```bash
uvicorn api.main:app --reload
```

The backend will run at:

`http://127.0.0.1:8000`

### Step 2: Start the Frontend

Open another terminal window, navigate to the project directory and execute:

```bash
python -m http.server 5500 --directory frontend
```

Open the application in your browser:

`http://localhost:5500`

### Step 3: API Documentation

FastAPI automatically provides interactive API documentation.

Open:

`http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks whether the API is running |
| GET | `/health` | Checks backend and model status |
| POST | `/predict` | Analyses billing data and returns anomaly results |
| POST | `/generate-report` | Generates a downloadable PDF report |

## Evaluation and Reports

The project generates evaluation outputs and reports, including:

- `legal_anomaly_results.csv` – Anomaly scores and classifications for test records.
- `flagged_legal_entries.csv` – Records identified as potential anomalies.
- `legal_anomaly_threshold.csv` – Selected anomaly threshold.
- `billing_characteristics_comparison.csv` – Comparison of normal and potential anomaly billing characteristics.
- `anomaly_scenario_comparison.csv` – Results of controlled anomaly experiments.
- `anomaly_scenario_comparison.png` – Scenario comparison visualization.

## Limitations

- The dataset does not contain verified fraud labels.
- The model identifies unusual billing patterns, not confirmed fraudulent transactions.
- Controlled anomaly experiments use artificially modified billing records.
- The threshold is based on the 95th percentile of validation reconstruction errors.
- Real-world fraud detection accuracy cannot be established without reliable labelled data.
- Human verification is required before making billing compliance decisions.

## Future Enhancements

- Evaluate the model using labelled legal billing datasets.
- Improve threshold selection using additional validation experiments.
- Introduce more advanced billing behaviour analysis.
- Expand the dashboard with additional analytics.
- Integrate with real-world legal billing systems.

## Conclusion

BillTimeGuard AI demonstrates how unsupervised machine learning can be applied to legal billing anomaly detection. By combining an Autoencoder model, reconstruction-error-based anomaly scoring, FastAPI integration and an interactive dashboard, the system helps identify potentially unusual billing patterns.

The project provides a decision-support approach that can help legal billing professionals prioritise records for further investigation and improve the efficiency of billing review.

**BillTimeGuard AI – Making Legal Billing Review Smarter with AI.**