# Data Preprocessing and Exploratory Data Analysis Streamlit Project

An interactive Streamlit application for end-to-end dataset exploration and preprocessing.

The app lets you upload a CSV dataset, inspect its structure, clean missing values, engineer features, encode categorical data, scale numeric columns, run PCA, visualize distributions/correlations, and export a processed CSV.

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
- Numeric scaling:
	- StandardScaler
	- MinMaxScaler
- PCA transformation with explained variance output
- Processed dataset download as CSV

## Project Structure

```text
Data-Preprocessing-and-Exploratory-Data-Analysis-Streamlit-Project/
├── app.py
└── README.md
```

## Requirements

- Python 3.9+
- pip

Python packages used:

- streamlit
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Installation

1. Open a terminal in the project folder.
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
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
