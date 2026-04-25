# Automated EDA, Preprocessing & Model Training (Streamlit)

An interactive Streamlit application for end-to-end dataset exploration, preprocessing, and machine learning workflows.

The app lets you upload a CSV dataset, profile data quality, choose between automatic or manual preprocessing, run exploratory analysis, train configurable ML models, and export a processed CSV.

## Features

- Robust CSV loading with fallback parsing attempts (encoding/engine retries)
- Dataset overview with preview, shape, columns, and dtypes
- Modernized UI with improved section layout, visual hierarchy, and updated app branding
- Automated dataset intelligence:
	- detailed column profile (dtype, missing %, unique count, sample value, suggested role)
	- suggested preprocessing plan
	- target candidates and model recommendations
- Two preprocessing modes:
	- **Automatic (One-Click):** duplicate removal, missing-value handling, date feature extraction, categorical encoding, numeric scaling, and downloadable processed output
	- **Manual (Step-by-Step):** full control of each preprocessing stage
- Data cleaning:
	- normalized column names
	- optional duplicate removal
	- optional column dropping
- Missing value handling by selected columns:
	- numeric: fill with mean, fill with median, or drop rows
	- categorical: fill with mode or drop rows
- Categorical encoding:
	- Label Encoding
	- One-Hot Encoding
- Feature engineering:
	- Date feature extraction (year, month, day, weekday, quarter, hour)
	- Combined feature creation with +, -, *, /
- Exploratory analysis:
	- summary statistics
	- top correlated feature pairs
	- correlation heatmap
	- numeric distribution plots (Histogram, KDE)
	- categorical count plots
- Scaling:
	- StandardScaler
	- MinMaxScaler
- PCA on selected numeric columns:
	- component choices limited to 2 or 3 (when data supports them)
	- 2 components -> 2D PCA scatter
	- 3 components -> 3D PCA scatter
	- explained variance ratio output
- Outlier detection:
	- IQR-based outlier summary
	- preview rows containing outliers
- Pre-training model intelligence:
	- analyzes processed dataset before training
	- recommends suitable model family and rationale
	- surfaces preprocessing suggestions before model configuration
- Modeling with configurable hyperparameters:
	- Classification: Logistic Regression, KNN, SVM, Decision Tree, Random Forest Classifier
	- Regression: Linear Regression, KNN, SVR, Decision Tree, Random Forest Regressor
	- Clustering: KMeans
- Automatic model visualizations:
	- classification: confusion matrix, feature influence, tree diagram or decision boundary
	- regression: residual plots, feature influence, tree diagram or regression value map
	- clustering: cluster scatter, cluster sizes, centroid heatmap
- Improved plot stability and compatibility fixes for count plots, decision boundaries, and colorbar rendering
- Pipeline summary of applied preprocessing steps
- Download processed dataset as CSV

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
2. Review **Dataset Overview** and **Automated Dataset Intelligence** outputs.
3. Choose a preprocessing path:
	- **Automatic (One-Click)** for quick end-to-end preprocessing
	- **Manual (Step-by-Step)** for granular control
4. In manual mode, optionally apply cleaning, missing-value handling, encoding, feature engineering, EDA, scaling/PCA, and outlier review.
5. Open **Model Selection & Training Analysis** to view dataset-driven model suggestions.
6. Train a model (classification, regression, or clustering) with selected hyperparameters.
7. Download the processed dataset as CSV.

## Notes

- Date columns are auto-detected from existing datetime columns and likely datetime-like text columns.
- PCA can run independently of scaling, but scaling is usually recommended for better component behavior.
- Automatic mode applies sensible defaults designed for quick baseline model training.
- Large datasets may make visualizations and model training slower to render.
