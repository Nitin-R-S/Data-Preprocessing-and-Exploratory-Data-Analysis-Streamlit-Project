# Data Preprocessing and Exploratory Data Analysis Streamlit Project

An interactive Streamlit application for end-to-end dataset exploration, preprocessing, and basic machine learning workflows.

The app lets you upload a CSV dataset, inspect its structure, clean missing values, engineer features, encode categorical data, scale numeric columns, run PCA, train baseline models, visualize patterns, and export a processed CSV.

## Features

- CSV upload and automatic dataset preview
- Column normalization and duplicate row removal
- Missing value handling for numeric and categorical columns
- Categorical encoding:
	- Label Encoding
	- One-Hot Encoding
- Feature engineering:
	- Date feature extraction (year, month, day, weekday, quarter, hour)
	- Combined feature creation using arithmetic operations
- Exploratory analysis:
	- Summary statistics
	- Correlation heatmap
	- Distribution plots (Histogram, KDE)
	- Count plots
	- Top correlated feature pairs
- Modeling support:
	- Classification (Logistic Regression, Random Forest, SVM)
	- Regression (Linear Regression, Random Forest Regressor)
	- Clustering (KMeans)
- Numeric scaling:
	- StandardScaler
	- MinMaxScaler
- PCA transformation with explained variance output
- Processed dataset download as CSV

## Project Structure

```text
Data-Preprocessing-and-Exploratory-Data-Analysis-Streamlit-Project/
├── app.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- pip
- Dependencies listed in requirements.txt:
	- streamlit==1.44.1
	- pandas==2.2.3
	- numpy==2.2.4
	- seaborn==0.13.2
	- matplotlib==3.10.1
	- scikit-learn==1.6.1

## Installation

1. Open a terminal and go to the project folder:

```bash
cd Data-Preprocessing-and-Exploratory-Data-Analysis-Streamlit-Project
```

2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the virtual environment:

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## How to Use

1. Upload a CSV file from the sidebar.
2. Review dataset preview, shape, columns, and dtypes.
3. Apply cleaning and missing-value handling options.
4. Encode categorical columns if needed.
5. Create engineered features.
6. Explore visualizations and summary statistics.
7. Apply scaling and optional PCA.
8. Download the processed dataset as CSV.

## Notes

- Date columns are auto-detected from existing datetime columns and likely datetime-like text columns.
- Large datasets may make pairwise plots and correlation charts slower to render.
