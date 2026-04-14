import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from typing import List, Tuple, Dict, Any

st.set_page_config(
    page_title="Automated EDA & Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin1")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    percent = 100 * missing / len(df)
    return pd.DataFrame({"missing_count": missing, "missing_percent": percent}).sort_values(
        by="missing_count", ascending=False
    )


def fill_missing_values(
    df: pd.DataFrame,
    numeric_method: str,
    categorical_method: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    df = df.copy()

    if numeric_method == "mean":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    elif numeric_method == "median":
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    if categorical_method == "mode":
        for col in categorical_cols:
            if df[col].isna().any():
                mode_value = df[col].mode(dropna=True)
                if not mode_value.empty:
                    df[col] = df[col].fillna(mode_value[0])
    return df


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
    object_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    low_cardinality = [col for col in detect_numeric_columns(df) if df[col].nunique() < 10]
    return sorted(list(set(object_cols + low_cardinality)))


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not date_cols:
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().head(20).astype(str)
                if sample.empty:
                    continue
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().sum() >= max(3, len(sample) // 2):
                        date_cols.append(col)
                except Exception:
                    continue
    return date_cols


def extract_date_features(df: pd.DataFrame, date_column: str, features: List[str]) -> pd.DataFrame:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    if df[date_column].isna().all():
        return df
    if "year" in features:
        df[f"{date_column}_year"] = df[date_column].dt.year
    if "month" in features:
        df[f"{date_column}_month"] = df[date_column].dt.month
    if "day" in features:
        df[f"{date_column}_day"] = df[date_column].dt.day
    if "weekday" in features:
        df[f"{date_column}_weekday"] = df[date_column].dt.day_name()
    if "quarter" in features:
        df[f"{date_column}_quarter"] = df[date_column].dt.quarter
    if "hour" in features:
        df[f"{date_column}_hour"] = df[date_column].dt.hour
    return df


def create_combined_feature(
    df: pd.DataFrame, new_column: str, left_column: str, right_column: str, operation: str
) -> pd.DataFrame:
    df = df.copy()
    if new_column in df.columns or new_column.strip() == "":
        return df
    if operation == "+":
        df[new_column] = df[left_column].astype(str) + df[right_column].astype(str)
    elif operation == "-":
        df[new_column] = df[left_column] - df[right_column]
    elif operation == "*":
        df[new_column] = df[left_column] * df[right_column]
    elif operation == "/":
        df[new_column] = df[left_column] / df[right_column].replace(0, np.nan)
    return df


def apply_custom_transformation(df: pd.DataFrame, expression: str) -> Tuple[pd.DataFrame, str]:
    df = df.copy()
    if "=" not in expression:
        return df, "Expression must contain an assignment operator (=)."
    try:
        target, expr = [part.strip() for part in expression.split("=", 1)]
        if not target:
            return df, "Provide a valid target variable name on the left side of =."
        local_vars = {col: df[col] for col in df.columns}
        local_vars.update({"pd": pd, "np": np})
        result = eval(expr, {}, local_vars)
        df[target] = result
        return df, f"Created '{target}' successfully."
    except Exception as exc:
        return df, f"Transformation failed: {exc}"


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").transpose()


def plot_correlation_heatmap(df: pd.DataFrame, size=(10, 8)) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are needed for correlation heatmap.")
        return
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)


def plot_distribution(df: pd.DataFrame, column: str, plot_type: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if plot_type == "Histogram":
        sns.histplot(df[column].dropna(), kde=False, ax=ax, color="#4c72b0")
        ax.set_title(f"Histogram of {column}")
    elif plot_type == "KDE":
        sns.kdeplot(df[column].dropna(), shade=True, ax=ax, color="#4c72b0")
        ax.set_title(f"KDE of {column}")
    st.pyplot(fig)


def plot_countplot(df: pd.DataFrame, column: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order, palette="viridis", ax=ax)
    ax.set_title(f"Count Plot for {column}")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


def plot_pairwise(df: pd.DataFrame, columns: List[str]) -> None:
    if len(columns) < 2:
        st.info("Select at least two columns for a pairplot.")
        return
    if len(columns) > 8:
        st.warning("Pairplot may be slow for many features; choose fewer columns.")
    fig = sns.pairplot(df[columns].dropna())
    st.pyplot(fig)


def encode_data(df: pd.DataFrame, method: str, categorical_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    if method == "Label Encoding":
        encoder = LabelEncoder()
        for col in categorical_cols:
            try:
                df[col] = encoder.fit_transform(df[col].astype(str))
            except Exception:
                continue
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def scale_data(df: pd.DataFrame, method: str, num_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler() if method == "StandardScaler" else MinMaxScaler()
    if num_cols:
        scaled_values = scaler.fit_transform(df[num_cols])
        df[num_cols] = scaled_values
    return df


def apply_pca(df: pd.DataFrame, n_components: int) -> Tuple[pd.DataFrame, np.ndarray]:
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(numeric_df)
    columns = [f"pca_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=columns, index=numeric_df.index)
    return pca_df, pca.explained_variance_ratio_


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_top_correlations(df: pd.DataFrame, top_n: int = 5) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("Not enough numeric columns to calculate top correlations.")
        return
    corr = numeric_df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False)
    corr = corr[corr != 1].drop_duplicates().head(top_n)
    st.table(corr.reset_index().rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_correlation"}))


def train_classification_model(df: pd.DataFrame, model_name: str, target_col: str, test_size: float = 0.2) -> Dict[str, Any]:
    """Train a classification model and return metrics."""
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove rows with missing values in target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            return {"error": "No valid data after removing missing values."}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "Random Forest Classifier":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "SVM":
            model = SVC(kernel="rbf", random_state=42)
        else:
            return {"error": "Unknown model name."}
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        return {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    except Exception as e:
        return {"error": str(e)}


def train_regression_model(df: pd.DataFrame, model_name: str, target_col: str, test_size: float = 0.2) -> Dict[str, Any]:
    """Train a regression model and return metrics."""
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove rows with missing values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            return {"error": "No valid data after removing missing values."}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            return {"error": "Unknown model name."}
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        return {
            "model": model,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
    except Exception as e:
        return {"error": str(e)}


def train_clustering_model(df: pd.DataFrame, n_clusters: int, numeric_cols: List[str]) -> Dict[str, Any]:
    """Train a KMeans clustering model and return metrics."""
    try:
        X = df[numeric_cols].dropna()
        
        if len(X) == 0:
            return {"error": "No valid data for clustering."}
        
        if len(X.columns) == 0:
            return {"error": "No numeric columns selected."}
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        inertia = model.inertia_
        silhouette_score = None
        
        return {
            "model": model,
            "clusters": clusters,
            "inertia": inertia,
            "n_samples": len(X),
            "n_clusters": n_clusters,
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    st.title("Automated EDA & Data Preprocessing App")
    st.markdown(
        "Use this app to upload a CSV dataset, explore it, clean it, engineer features, encode and scale variables, apply PCA, and download the processed data."
    )

    sidebar = st.sidebar
    sidebar.header("Upload and Settings")
    uploaded_file = sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin automated exploratory data analysis and preprocessing.")
        st.stop()

    with st.spinner("Loading dataset..."):
        original_df = load_data(uploaded_file)

    if "df" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state.df = normalize_column_names(original_df.copy())
        st.session_state.uploaded_file_name = uploaded_file.name
    df = st.session_state.df
    pipeline_steps = ["Normalized column names"]

    st.subheader("Dataset Overview")
    st.write("**Preview of the uploaded dataset**")
    st.dataframe(df.head(10))
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data types:**")
    st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]))

    st.markdown("---")
    st.header("Missing Values Before Cleaning")
    original_missing_summary = get_missing_summary(df)
    if original_missing_summary["missing_count"].sum() == 0:
        st.success("No missing values detected in the uploaded dataset.")
    else:
        st.write("Review missing values before applying cleaning steps.")
        st.dataframe(original_missing_summary)

    st.markdown("---")
    st.header("Data Cleaning")
    clean_cols = st.multiselect("Drop columns", options=df.columns.tolist(), help="Remove irrelevant columns from the dataset.")
    remove_duplicates = st.checkbox("Remove duplicate rows", value=True)

    if remove_duplicates:
        with st.spinner("Removing duplicates..."):
            df = remove_duplicate_rows(df)
        pipeline_steps.append("Removed duplicate rows")

    if clean_cols:
        with st.spinner("Dropping selected columns..."):
            df = df.drop(columns=clean_cols, errors="ignore")
        pipeline_steps.append(f"Dropped columns: {', '.join(clean_cols)}")

    st.write("**After cleaning shape:**", df.shape)

    numeric_columns = detect_numeric_columns(df)
    categorical_columns = detect_categorical_columns(df)

    st.subheader("Handle Missing Values")
    missing_summary = get_missing_summary(df)
    st.write("### Missing values by column")
    st.dataframe(missing_summary)

    missing_numeric_cols = [col for col in numeric_columns if df[col].isna().sum() > 0]
    missing_categorical_cols = [col for col in categorical_columns if df[col].isna().sum() > 0]

    if missing_numeric_cols or missing_categorical_cols:
        if missing_numeric_cols:
            st.subheader("Numeric missing values")
            selected_numeric_cols = st.multiselect(
                "Select numeric columns to handle",
                options=missing_numeric_cols,
            )
            numeric_strategy = st.radio(
                "Numeric strategy",
                options=["Fill with Mean", "Fill with Median", "Drop rows"],
                key="numeric_strategy",
            )
        else:
            selected_numeric_cols = []
            numeric_strategy = None

        if missing_categorical_cols:
            st.subheader("Categorical missing values")
            selected_categorical_cols = st.multiselect(
                "Select categorical columns to handle",
                options=missing_categorical_cols,
            )
            categorical_strategy = st.radio(
                "Categorical strategy",
                options=["Fill with Mode", "Drop rows"],
                key="categorical_strategy",
            )
        else:
            selected_categorical_cols = []
            categorical_strategy = None

        if st.button("Apply Changes"):
            changes_applied = False
            if selected_numeric_cols and numeric_strategy:
                if numeric_strategy == "Fill with Mean":
                    for col in selected_numeric_cols:
                        df[col] = df[col].fillna(df[col].mean())
                    st.success(f"Filled missing values in numeric columns {', '.join(selected_numeric_cols)} with mean.")
                elif numeric_strategy == "Fill with Median":
                    for col in selected_numeric_cols:
                        df[col] = df[col].fillna(df[col].median())
                    st.success(f"Filled missing values in numeric columns {', '.join(selected_numeric_cols)} with median.")
                else:
                    df = df.dropna(subset=selected_numeric_cols).reset_index(drop=True)
                    st.success(f"Dropped rows with missing values in numeric columns {', '.join(selected_numeric_cols)}.")
                changes_applied = True

            if selected_categorical_cols and categorical_strategy:
                if categorical_strategy == "Fill with Mode":
                    filled_cols = []
                    for col in selected_categorical_cols:
                        mode_value = df[col].mode(dropna=True)
                        if not mode_value.empty:
                            df[col] = df[col].fillna(mode_value[0])
                            filled_cols.append(col)
                    if filled_cols:
                        st.success(f"Filled missing values in categorical columns {', '.join(filled_cols)} with mode.")
                        changes_applied = True
                    else:
                        st.warning("No mode value available for the selected categorical columns.")
                else:
                    df = df.dropna(subset=selected_categorical_cols).reset_index(drop=True)
                    st.success(f"Dropped rows with missing values in categorical columns {', '.join(selected_categorical_cols)}.")
                    changes_applied = True

            if changes_applied:
                st.session_state.df = df
                st.write("**Updated dataset after missing value handling:**")
                st.dataframe(df.head(10))
    else:
        st.info("No missing values found in numeric or categorical columns.")

    st.markdown("---")
    st.header("Encoding")
    encoding_method = st.selectbox(
        "Encoding method for categorical columns",
        options=["None", "Label Encoding", "One-Hot Encoding"],
        index=0,
    )

    if encoding_method != "None":
        selected_encoding_cols = st.multiselect(
            "Select categorical columns to encode",
            options=categorical_columns,
            default=[],
            help="Choose one or more categorical columns to encode.",
        )
        if st.button("Apply Encoding"):
            if selected_encoding_cols:
                with st.spinner("Applying encoding..."):
                    df = encode_data(df, encoding_method, selected_encoding_cols)
                st.session_state.df = df
                pipeline_steps.append(f"Applied {encoding_method} to {', '.join(selected_encoding_cols)}")
                st.success(f"Applied {encoding_method} to {', '.join(selected_encoding_cols)}.")
                st.write("**Updated dataset after encoding:**")
                st.dataframe(df.head(10))
            else:
                st.warning("Select categorical columns to encode before applying.")
    else:
        selected_encoding_cols = []

    st.markdown("---")
    st.header("Feature Engineering")
    date_cols = detect_date_columns(df)
    if date_cols:
        st.write("Detected date columns:", date_cols)
        selected_date = st.selectbox("Select a date column to extract features", options=date_cols)
        date_features = st.multiselect(
            "Select date features to extract",
            options=["year", "month", "day", "weekday", "quarter", "hour"],
            default=["year", "month", "day"],
        )
        if selected_date and date_features:
            with st.spinner("Extracting date features..."):
                df = extract_date_features(df, selected_date, date_features)
            st.session_state.df = df
            pipeline_steps.append(f"Extracted date features from {selected_date}")

    st.subheader("Create new features")
    with st.expander("Column combination feature"):
        new_feature_name = st.text_input("New feature name", value="")
        left_column = st.selectbox("Left column", options=df.columns.tolist(), key="left_col")
        right_column = st.selectbox("Right column", options=df.columns.tolist(), key="right_col")
        operation = st.selectbox("Operation", options=["+", "-", "*", "/"])
        if st.button("Create combined feature") and new_feature_name:
            with st.spinner("Creating new feature..."):
                df = create_combined_feature(df, new_feature_name, left_column, right_column, operation)
            st.session_state.df = df
            pipeline_steps.append(f"Created feature {new_feature_name} = {left_column} {operation} {right_column}")
            st.success(f"Feature '{new_feature_name}' created.")

    st.markdown("---")
    st.header("Exploratory Data Analysis")
    if st.checkbox("Show summary statistics", value=True):
        with st.spinner("Computing summary statistics..."):
            st.dataframe(generate_summary_statistics(df))

    if st.checkbox("Show top correlated feature pairs", value=True):
        with st.spinner("Calculating top correlations..."):
            show_top_correlations(df, top_n=5)

    if st.checkbox("Show correlation heatmap", value=True):
        with st.spinner("Rendering correlation heatmap..."):
            plot_correlation_heatmap(df)

    dist_column = st.selectbox("Select numeric column for distribution plot", options=numeric_columns, index=0 if numeric_columns else None)
    plot_type = st.selectbox("Distribution plot type", options=["Histogram", "KDE"])
    if dist_column and st.button("Show distribution plot"):
        plot_distribution(df, dist_column, plot_type)

    if categorical_columns:
        cat_col_for_count = st.selectbox("Select categorical column for count plot", options=categorical_columns)
        if st.button("Show count plot"):
            plot_countplot(df, cat_col_for_count)

    st.markdown("---")
    st.header("Scaling")
    scaling_method = st.selectbox(
        "Scaling method for numeric columns",
        options=["None", "StandardScaler", "MinMaxScaler"],
        index=0,
    )
    pca_enabled = st.checkbox("Apply PCA", value=False)
    n_components = st.slider("Number of PCA components", min_value=2, max_value=min(10, max(2, len(detect_numeric_columns(df)))), value=2)

    processed_df = df.copy()
    numeric_columns = detect_numeric_columns(processed_df)

    if st.button("Apply Scaling"):
        if scaling_method != "None" and numeric_columns:
            with st.spinner("Applying scaling..."):
                processed_df = scale_data(processed_df, scaling_method, numeric_columns)
            pipeline_steps.append(f"Applied {scaling_method}")
            st.session_state.df = processed_df
            st.success(f"Applied {scaling_method} to numeric columns.")
        else:
            st.info("No numeric columns available to scale or scaling method is None.")

    if pca_enabled and numeric_columns:
        with st.spinner("Running PCA..."):
            pca_df, variance = apply_pca(processed_df, n_components)
        st.subheader("PCA Results")
        st.write("Explained variance ratio:")
        st.write(variance.round(4))
        if pca_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=pca_df.iloc[:, 0], y=pca_df.iloc[:, 1], ax=ax)
            ax.set_xlabel("PCA component 1")
            ax.set_ylabel("PCA component 2")
            ax.set_title("PCA 2D Scatter Plot")
            st.pyplot(fig)
        pipeline_steps.append(f"Applied PCA with {n_components} components")
        processed_df = pd.concat([processed_df.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    st.write("Preview of the processed dataset after applied transformations.")
    st.dataframe(processed_df.head(20))
    st.write("**Processed shape:**", processed_df.shape)

    csv_data = dataframe_to_csv(processed_df)
    st.download_button(
        label="Download processed dataset as CSV",
        data=csv_data,
        file_name="processed_dataset.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.header("Model Training")
    
    model_type = st.selectbox(
        "Select Model Type",
        options=["Classification", "Regression", "Clustering"],
        index=0,
    )
    
    if model_type == "Classification":
        st.subheader("Classification Models")
        st.write("Train a classification model to predict categorical targets.")
        
        numeric_cols = detect_numeric_columns(processed_df)
        all_cols = processed_df.columns.tolist()
        
        target_col = st.selectbox(
            "Select target column (must be numeric or encoded categorical)",
            options=all_cols,
            key="class_target",
        )
        
        model_name = st.selectbox(
            "Select Classification Model",
            options=["Logistic Regression", "Random Forest Classifier", "SVM"],
            key="class_model",
        )
        
        test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="class_test")
        
        if st.button("Train Classification Model"):
            with st.spinner(f"Training {model_name}..."):
                results = train_classification_model(processed_df, model_name, target_col, test_size)
            
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                st.success(f"Model trained successfully!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{results['accuracy']:.4f}")
                col2.metric("Precision", f"{results['precision']:.4f}")
                col3.metric("Recall", f"{results['recall']:.4f}")
                col4.metric("F1-Score", f"{results['f1_score']:.4f}")
                
                st.write(f"**Training set size:** {results['train_size']}")
                st.write(f"**Test set size:** {results['test_size']}")
    
    elif model_type == "Regression":
        st.subheader("Regression Models")
        st.write("Train a regression model to predict continuous targets.")
        
        numeric_cols = detect_numeric_columns(processed_df)
        
        if len(numeric_cols) < 2:
            st.warning("At least 2 numeric columns are required for regression (features + target).")
        else:
            target_col = st.selectbox(
                "Select target column (must be numeric)",
                options=numeric_cols,
                key="reg_target",
            )
            
            model_name = st.selectbox(
                "Select Regression Model",
                options=["Linear Regression", "Random Forest Regressor"],
                key="reg_model",
            )
            
            test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="reg_test")
            
            if st.button("Train Regression Model"):
                with st.spinner(f"Training {model_name}..."):
                    results = train_regression_model(processed_df, model_name, target_col, test_size)
                
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    st.success(f"Model trained successfully!")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R² Score", f"{results['r2_score']:.4f}")
                    col2.metric("RMSE", f"{results['rmse']:.4f}")
                    col3.metric("MSE", f"{results['mse']:.4f}")
                    
                    st.write(f"**Training set size:** {results['train_size']}")
                    st.write(f"**Test set size:** {results['test_size']}")
    
    elif model_type == "Clustering":
        st.subheader("Clustering Models")
        st.write("Train a KMeans clustering model to group similar data points.")
        
        numeric_cols = detect_numeric_columns(processed_df)
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for clustering.")
        else:
            selected_cluster_cols = st.multiselect(
                "Select numeric columns for clustering",
                options=numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))],
                key="cluster_cols",
            )
            
            n_clusters = st.slider(
                "Number of clusters",
                min_value=2,
                max_value=min(10, len(processed_df)),
                value=3,
                key="n_clusters",
            )
            
            if st.button("Train Clustering Model"):
                if not selected_cluster_cols:
                    st.warning("Select at least one numeric column for clustering.")
                else:
                    with st.spinner("Training KMeans..."):
                        results = train_clustering_model(processed_df, n_clusters, selected_cluster_cols)
                    
                    if "error" in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.success("KMeans clustering completed successfully!")
                        col1, col2 = st.columns(2)
                        col1.metric("Number of Clusters", results['n_clusters'])
                        col2.metric("Inertia (WCSS)", f"{results['inertia']:.4f}")
                        
                        st.write(f"**Number of samples clustered:** {results['n_samples']}")
                        
                        # Visualize clusters if we have 2 or more dimensions
                        if len(selected_cluster_cols) >= 2:
                            st.subheader("Cluster Visualization")
                            X_cluster = processed_df[selected_cluster_cols].dropna()
                            clusters = results['clusters'][:len(X_cluster)]
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            scatter = ax.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=clusters, cmap="viridis", alpha=0.6)
                            ax.set_xlabel(selected_cluster_cols[0])
                            ax.set_ylabel(selected_cluster_cols[1])
                            ax.set_title(f"KMeans Clustering (k={n_clusters})")
                            plt.colorbar(scatter, ax=ax, label="Cluster")
                            st.pyplot(fig)

    st.markdown("---")
    st.header("Pipeline Summary")
    if pipeline_steps:
        for i, step in enumerate(pipeline_steps, start=1):
            st.write(f"{i}. {step}")
    else:
        st.write("No preprocessing steps were applied yet.")

    st.markdown("---")
    st.header("Notes")
    st.info(
        "This app uses pandas, numpy, seaborn, matplotlib and scikit-learn to provide a full data cleaning and exploration workflow." 
        "If your dataset has datetime-like text columns, the app will detect them and offer date feature extraction."
    )


if __name__ == "__main__":
    main()
