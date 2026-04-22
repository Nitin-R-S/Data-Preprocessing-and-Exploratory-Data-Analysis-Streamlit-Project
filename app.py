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
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
from typing import List, Tuple, Dict, Any

st.set_page_config(
    page_title="Automated EDA & Preprocessing",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_ui_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-main: #f3f4ef;
            --surface: #ffffff;
            --surface-soft: #f9fbf8;
            --text-main: #1f2a22;
            --text-muted: #5c6a60;
            --brand: #245f45;
            --brand-strong: #143b2a;
            --brand-soft: #d9f0e4;
            --accent: #d9683f;
            --accent-soft: #ffeadf;
            --line: #dde6df;
            --shadow-soft: 0 10px 24px rgba(22, 44, 32, 0.1);
            --shadow-card: 0 16px 36px rgba(22, 44, 32, 0.12);
            --font-body: "Aptos", "Segoe UI Variable", "Segoe UI", "Trebuchet MS", sans-serif;
            --font-display: "Bahnschrift", "Franklin Gothic Medium", "Segoe UI Semibold", "Arial Narrow", sans-serif;
        }

        @keyframes fadeUp {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes glowShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 12%, #edf8f1 0%, transparent 38%),
                radial-gradient(circle at 88% 0%, #ffe9de 0%, transparent 35%),
                var(--bg-main);
            color: var(--text-main);
            font-family: var(--font-body);
            animation: fadeUp 0.45s ease-out;
        }

        .stApp,
        .stApp p,
        .stApp label,
        .stApp li,
        .stApp a,
        .stApp th,
        .stApp td,
        .stApp input,
        .stApp textarea,
        .stApp select,
        .stApp button {
            font-family: var(--font-body);
        }

        /* Keep Streamlit icon ligatures on Material icon fonts. */
        .material-icons,
        .material-icons-round,
        .material-icons-outlined,
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-symbols-sharp,
        [class*="material-icons"],
        [class*="material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
            font-style: normal;
            font-weight: normal;
            letter-spacing: normal;
            text-transform: none;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
            -webkit-font-feature-settings: "liga";
            font-feature-settings: "liga";
        }

        *, *::before, *::after {
            box-sizing: border-box;
        }

        .block-container {
            padding-top: 1.4rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            font-family: var(--font-display);
            letter-spacing: -0.01em;
            color: var(--text-main);
        }

        h1, h2, h3, .hero-title, div[data-testid="stMetricValue"] {
            font-family: var(--font-display) !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f3b2c 0%, #102019 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            overflow-x: hidden;
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
            padding-left: 0.95rem;
            padding-right: 0.95rem;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] li,
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] small,
        section[data-testid="stSidebar"] span {
            color: #ecf4ef;
        }

        section[data-testid="stSidebar"] .stFileUploader > div {
            border: 1px dashed rgba(236, 244, 239, 0.55);
            border-radius: 12px;
            background-color: rgba(255, 255, 255, 0.03);
            padding: 0.3rem;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
            background-color: rgba(255, 255, 255, 0.06);
            border: 1px dashed rgba(236, 244, 239, 0.5);
        }

        section[data-testid="stSidebar"] .stFileUploader button {
            color: #f4fbf7 !important;
            border: 1px solid rgba(236, 244, 239, 0.55) !important;
            background: rgba(255, 255, 255, 0.1) !important;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        section[data-testid="stSidebar"] .stFileUploader button:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            border-color: rgba(236, 244, 239, 0.8) !important;
        }

        section[data-testid="stSidebar"] .stFileUploader small {
            color: #cce0d3 !important;
        }

        .sidebar-label {
            margin: 0.25rem 0 0.45rem;
            color: #eff7f2;
            font-family: var(--font-display);
            font-size: 1.05rem;
            letter-spacing: 0.01em;
        }

        .sidebar-nav-wrap {
            margin-top: 0.4rem;
            border: 1px solid rgba(223, 239, 229, 0.22);
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.03) 100%);
            border-radius: 14px;
            padding: 0.65rem;
        }

        .sidebar-nav-title {
            margin: 0.05rem 0 0.45rem;
            color: #f5fbf7;
            font-family: var(--font-display);
            font-size: 1rem;
            letter-spacing: 0.01em;
        }

        .sidebar-nav-list {
            display: grid;
            grid-template-columns: 1fr;
            gap: 0.3rem;
        }

        .sidebar-nav-link {
            display: block;
            color: #e4f2e9 !important;
            text-decoration: none !important;
            border: 1px solid rgba(223, 239, 229, 0.18);
            border-radius: 10px;
            padding: 0.36rem 0.55rem;
            font-size: 0.9rem;
            line-height: 1.25;
            background: rgba(255, 255, 255, 0.03);
            transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
        }

        .sidebar-nav-link:hover {
            background: rgba(217, 240, 228, 0.16);
            border-color: rgba(223, 239, 229, 0.45);
            transform: translateX(2px);
        }

        .sidebar-nav-link::before {
            content: "#";
            display: inline-block;
            color: #9dd5b8;
            margin-right: 0.36rem;
            font-weight: 700;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
            color: #d6e6db !important;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * {
            max-width: 100%;
            overflow-wrap: anywhere;
            word-break: break-word;
        }

        .hero-shell {
            background: linear-gradient(130deg, #245f45 0%, #143b2a 45%, #0f291e 100%);
            background-size: 200% 200%;
            color: #f2f8f4;
            border-radius: 20px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.1rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 18px 40px rgba(24, 44, 33, 0.3);
            position: relative;
            overflow: hidden;
            animation: glowShift 14s ease-in-out infinite;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -20% -70% 45%;
            width: 320px;
            height: 320px;
            background: radial-gradient(circle, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0) 70%);
            pointer-events: none;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(1.3rem, 1.8vw, 2rem);
            font-weight: 700;
            color: #f6fbf8;
        }

        .hero-subtitle {
            margin-top: 0.45rem;
            margin-bottom: 0;
            color: #d3e9dc;
            line-height: 1.45;
            font-size: 0.95rem;
        }

        .hero-meta {
            margin-top: 0.8rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            position: relative;
            z-index: 1;
        }

        .hero-chip {
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #f3fbf6;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.74rem;
            letter-spacing: 0.02em;
            backdrop-filter: blur(2px);
        }

        .section-title-wrap {
            margin-top: 0.3rem;
            margin-bottom: 0.7rem;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.8rem 0.95rem;
            box-shadow: var(--shadow-soft);
            border-left: 5px solid var(--accent);
        }

        .section-kicker {
            color: var(--accent);
            font-weight: 800;
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .section-title {
            margin: 0.2rem 0 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .stDataFrame, .stTable {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: var(--surface);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f5faf7 100%);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 0.45rem 0.65rem;
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stMetricValue"] {
            font-family: var(--font-display);
            color: #1e5038;
            letter-spacing: -0.02em;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #1e5038;
            background: linear-gradient(180deg, #2d7656 0%, #1b4d37 100%);
            color: #f4fbf7;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(30, 80, 56, 0.2);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #153a29;
            background: linear-gradient(180deg, #1f5a40 0%, #174632 100%);
            color: #ffffff;
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(30, 80, 56, 0.25);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: #eef3ef;
            border-radius: 12px;
            padding: 0.25rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            font-weight: 600;
            color: #355243;
            padding: 0.45rem 0.85rem;
        }

        .stTabs [aria-selected="true"] {
            background: #ffffff;
            color: var(--brand-strong);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: var(--surface-soft);
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stTextInput"] input,
        div[data-testid="stVerticalBlock"] div[data-testid="stNumberInput"] input,
        div[data-testid="stVerticalBlock"] div[data-testid="stSelectbox"] > div,
        div[data-testid="stVerticalBlock"] div[data-testid="stMultiSelect"] > div,
        div[data-testid="stVerticalBlock"] div[data-testid="stTextArea"] textarea {
            border-radius: 10px;
            border-color: #cedbd1;
        }

        div[data-testid="stVerticalBlock"] input:focus,
        div[data-testid="stVerticalBlock"] textarea:focus {
            border-color: #2c6c4f !important;
            box-shadow: 0 0 0 2px rgba(44, 108, 79, 0.15);
        }

        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }

            .hero-shell {
                padding: 1rem;
                border-radius: 14px;
            }

            .hero-meta {
                gap: 0.35rem;
            }

            .hero-chip {
                font-size: 0.69rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str) -> None:
    st.markdown(
        f"""
        <div class="section-title-wrap">
            <div class="section-kicker">Workflow Section</div>
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    read_attempts = [
        {"encoding": None, "sep": None, "engine": "c"},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "latin1", "sep": None, "engine": "python"},
        {"encoding": "utf-8", "sep": None, "engine": "python", "on_bad_lines": "skip"},
        {"encoding": "latin1", "sep": None, "engine": "python", "on_bad_lines": "skip"},
    ]

    last_error = None
    for attempt in read_attempts:
        try:
            uploaded_file.seek(0)
            params = {"engine": attempt["engine"]}
            if attempt["encoding"] is not None:
                params["encoding"] = attempt["encoding"]
            if "on_bad_lines" in attempt:
                params["on_bad_lines"] = attempt["on_bad_lines"]
            return pd.read_csv(uploaded_file, **params)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Unable to parse the uploaded CSV file. Details: {last_error}")


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
        if not (pd.api.types.is_numeric_dtype(df[left_column]) and pd.api.types.is_numeric_dtype(df[right_column])):
            return df
        df[new_column] = df[left_column] - df[right_column]
    elif operation == "*":
        if not (pd.api.types.is_numeric_dtype(df[left_column]) and pd.api.types.is_numeric_dtype(df[right_column])):
            return df
        df[new_column] = df[left_column] * df[right_column]
    elif operation == "/":
        if not (pd.api.types.is_numeric_dtype(df[left_column]) and pd.api.types.is_numeric_dtype(df[right_column])):
            return df
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


def render_figure(fig: plt.Figure) -> None:
    try:
        try:
            st.pyplot(fig, width="stretch", clear_figure=True)
        except TypeError:
            st.pyplot(fig, use_container_width=True, clear_figure=True)
    except TypeError:
        st.pyplot(fig, clear_figure=True)
    finally:
        plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, size=(10, 8)) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        st.info("At least two numeric columns are needed for correlation heatmap.")
        return
    try:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=size)
        fig.patch.set_facecolor("white")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
        ax.set_title("Correlation Matrix")
        render_figure(fig)
    except Exception as exc:
        st.warning(f"Could not render heatmap: {exc}")


def plot_distribution(df: pd.DataFrame, column: str, plot_type: str) -> None:
    try:
        values = df[column].dropna()
        if values.empty:
            st.info(f"No non-missing values available for {column}.")
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        if plot_type == "Histogram":
            sns.histplot(values, kde=False, ax=ax, color="#4c72b0")
            ax.set_title(f"Histogram of {column}")
        elif plot_type == "KDE":
            sns.kdeplot(values, fill=True, ax=ax, color="#4c72b0")
            ax.set_title(f"KDE of {column}")
        render_figure(fig)
    except Exception as exc:
        st.warning(f"Could not render distribution plot: {exc}")


def plot_countplot(df: pd.DataFrame, column: str) -> None:
    try:
        order = df[column].dropna().value_counts().index
        if len(order) == 0:
            st.info(f"No non-missing values available for {column}.")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("white")
        sns.countplot(data=df, x=column, order=order, palette="viridis", ax=ax)
        ax.set_title(f"Count Plot for {column}")
        plt.xticks(rotation=45, ha="right")
        render_figure(fig)
    except Exception as exc:
        st.warning(f"Could not render count plot: {exc}")


def plot_pairwise(df: pd.DataFrame, columns: List[str]) -> None:
    if len(columns) < 2:
        st.info("Select at least two columns for a pairplot.")
        return
    if len(columns) > 8:
        st.warning("Pairplot may be slow for many features; choose fewer columns.")
    try:
        pair_data = df[columns].dropna()
        if pair_data.empty:
            st.info("No rows available for pairplot after removing missing values.")
            return
        pair_grid = sns.pairplot(pair_data)
        render_figure(pair_grid.figure)
    except Exception as exc:
        st.warning(f"Could not render pairplot: {exc}")


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


def apply_pca(df: pd.DataFrame, n_components: int, selected_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, str]:
    if not selected_cols:
        return pd.DataFrame(), np.array([]), "Select at least one numeric column for PCA."

    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        return pd.DataFrame(), np.array([]), f"Selected PCA columns not found: {', '.join(missing_cols)}"

    non_numeric_cols = [col for col in selected_cols if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric_cols:
        return pd.DataFrame(), np.array([]), f"PCA columns must be numeric: {', '.join(non_numeric_cols)}"

    numeric_df = df[selected_cols].dropna()
    if numeric_df.empty:
        return pd.DataFrame(), np.array([]), "PCA requires rows without missing values in selected columns."

    max_components = min(numeric_df.shape[0], numeric_df.shape[1])
    if max_components < 1:
        return pd.DataFrame(), np.array([]), "PCA requires at least one numeric feature."

    if n_components > max_components:
        return (
            pd.DataFrame(),
            np.array([]),
            f"Requested {n_components} components, but only {max_components} are possible for the current data.",
        )

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(numeric_df)
    columns = [f"pca_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=columns, index=numeric_df.index)
    return pca_df, pca.explained_variance_ratio_, ""


def plot_pca_projection(pca_df: pd.DataFrame, variance: np.ndarray) -> None:
    if pca_df.shape[1] < 2:
        st.info("PCA visualization requires at least two components.")
        return

    component_x = pca_df.columns[0]
    component_y = pca_df.columns[1]
    explained_x = variance[0] * 100 if len(variance) > 0 else None
    explained_y = variance[1] * 100 if len(variance) > 1 else None

    if pca_df.shape[1] == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=pca_df,
            x=component_x,
            y=component_y,
            ax=ax,
            s=55,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.4,
            color="#245f45",
        )
        ax.set_title("PCA 2D Scatter Plot")
        ax.set_xlabel(f"{component_x} ({explained_x:.1f}% variance)" if explained_x is not None else component_x)
        ax.set_ylabel(f"{component_y} ({explained_y:.1f}% variance)" if explained_y is not None else component_y)
        ax.axhline(0, color="#b7c6bb", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="#b7c6bb", linewidth=0.8, linestyle="--")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)
        fig.tight_layout()
        render_figure(fig)
        return

    component_z = pca_df.columns[2]
    explained_z = variance[2] * 100 if len(variance) > 2 else None
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pca_df[component_x],
        pca_df[component_y],
        pca_df[component_z],
        s=45,
        alpha=0.78,
        c="#245f45",
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_title("PCA 3D Scatter Plot")
    ax.set_xlabel(f"{component_x} ({explained_x:.1f}% variance)" if explained_x is not None else component_x)
    ax.set_ylabel(f"{component_y} ({explained_y:.1f}% variance)" if explained_y is not None else component_y)
    ax.set_zlabel(f"{component_z} ({explained_z:.1f}% variance)" if explained_z is not None else component_z)
    render_figure(fig)


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


def summarize_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_cols = detect_numeric_columns(df)
    if not numeric_cols:
        return pd.DataFrame(), pd.Series([False] * len(df), index=df.index)

    summary_rows = []
    row_outlier_mask = pd.Series([False] * len(df), index=df.index)

    for col in numeric_cols:
        series = df[col]
        non_null = series.dropna()
        if len(non_null) < 4:
            continue

        q1 = non_null.quantile(0.25)
        q3 = non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        col_mask = (series < lower) | (series > upper)
        count = int(col_mask.sum())
        if count == 0:
            continue

        row_outlier_mask = row_outlier_mask | col_mask.fillna(False)
        summary_rows.append(
            {
                "column": col,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": count,
                "outlier_percent": round((count / len(df) * 100), 2) if len(df) else 0.0,
            }
        )

    if not summary_rows:
        return pd.DataFrame(), row_outlier_mask

    summary_df = pd.DataFrame(summary_rows).sort_values(by="outlier_count", ascending=False).reset_index(drop=True)
    return summary_df, row_outlier_mask


def build_column_profile(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        col_data = df[col]
        missing_count = int(col_data.isna().sum())
        unique_count = int(col_data.nunique(dropna=True))
        non_null = col_data.dropna()
        sample_value = str(non_null.iloc[0]) if not non_null.empty else "N/A"
        role = "feature"
        if col in date_cols:
            role = "date_feature_candidate"
        elif pd.api.types.is_numeric_dtype(col_data):
            if 2 <= unique_count <= 10:
                role = "classification_target_candidate"
            elif unique_count > 10:
                role = "regression_target_candidate"
        else:
            if 2 <= unique_count <= 20:
                role = "classification_target_candidate"

        rows.append(
            {
                "column": col,
                "dtype": str(col_data.dtype),
                "missing_count": missing_count,
                "missing_percent": round((missing_count / len(df) * 100), 2) if len(df) else 0.0,
                "unique_count": unique_count,
                "sample_value": sample_value,
                "suggested_role": role,
            }
        )

    return pd.DataFrame(rows).sort_values(by=["missing_percent", "unique_count"], ascending=[False, False])


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = detect_numeric_columns(df)
    categorical_cols = detect_categorical_columns(df)
    date_cols = detect_date_columns(df)

    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_percent_total = round((missing_cells / total_cells * 100), 2) if total_cells else 0.0

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
    high_cardinality_cols = [
        col
        for col in categorical_cols
        if df[col].nunique(dropna=True) > max(50, int(0.5 * len(df)))
    ]

    skewed_cols = []
    for col in numeric_cols:
        col_non_null = df[col].dropna()
        if len(col_non_null) >= 20:
            try:
                skewness = float(col_non_null.skew())
                if abs(skewness) > 1:
                    skewed_cols.append((col, skewness))
            except Exception:
                continue

    outlier_cols = []
    for col in numeric_cols:
        col_non_null = df[col].dropna()
        if len(col_non_null) < 20:
            continue
        q1 = col_non_null.quantile(0.25)
        q3 = col_non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_ratio = ((col_non_null < lower) | (col_non_null > upper)).mean()
        if outlier_ratio > 0.05:
            outlier_cols.append((col, float(outlier_ratio)))

    classification_targets = [
        col for col in df.columns if 2 <= df[col].nunique(dropna=True) <= 20 and df[col].dropna().shape[0] >= 20
    ]
    regression_targets = [
        col for col in numeric_cols if df[col].nunique(dropna=True) > 20 and df[col].dropna().shape[0] >= 20
    ]

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "date_cols": date_cols,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "missing_percent_total": missing_percent_total,
        "constant_cols": constant_cols,
        "high_cardinality_cols": high_cardinality_cols,
        "skewed_cols": skewed_cols,
        "outlier_cols": outlier_cols,
        "classification_targets": classification_targets,
        "regression_targets": regression_targets,
        "column_profile": build_column_profile(df, date_cols),
    }


def preprocessing_suggestions(analysis: Dict[str, Any]) -> List[str]:
    suggestions = []
    if analysis["duplicate_rows"] > 0:
        suggestions.append(f"Remove {analysis['duplicate_rows']} duplicate rows.")
    if analysis["missing_cells"] > 0:
        suggestions.append(
            f"Handle missing values ({analysis['missing_percent_total']}% of all cells): median/mean for numeric and mode or separate 'missing' class for categorical columns."
        )
    if analysis["constant_cols"]:
        suggestions.append(f"Drop constant columns: {', '.join(analysis['constant_cols'][:8])}.")
    if analysis["high_cardinality_cols"]:
        suggestions.append(
            f"High-cardinality categorical columns detected ({', '.join(analysis['high_cardinality_cols'][:6])}); prefer target/frequency encoding or grouping rare categories."
        )
    if analysis["date_cols"]:
        suggestions.append(
            f"Extract date features from: {', '.join(analysis['date_cols'][:6])} (year, month, weekday, quarter)."
        )
    if analysis["skewed_cols"]:
        skew_cols = [item[0] for item in analysis["skewed_cols"][:6]]
        suggestions.append(f"Skewed numeric features found ({', '.join(skew_cols)}); consider log or Box-Cox style transforms.")
    if analysis["outlier_cols"]:
        out_cols = [item[0] for item in analysis["outlier_cols"][:6]]
        suggestions.append(f"Potential outliers in ({', '.join(out_cols)}); consider clipping/winsorizing or robust models.")
    if analysis["numeric_cols"]:
        suggestions.append("Scale numeric features before SVM, SVR, KMeans, and PCA.")
    if not suggestions:
        suggestions.append("Dataset is relatively clean; proceed with light preprocessing and baseline model training.")
    return suggestions


def model_recommendations(analysis: Dict[str, Any]) -> Dict[str, Any]:
    n_rows = analysis["rows"]
    has_numeric = len(analysis["numeric_cols"]) > 0
    has_classification = len(analysis["classification_targets"]) > 0
    has_regression = len(analysis["regression_targets"]) > 0

    recommendation_lines = []
    if has_classification:
        if n_rows >= 500:
            recommendation_lines.append("Classification: Random Forest Classifier is a strong default baseline for mixed and non-linear data.")
        else:
            recommendation_lines.append("Classification: Logistic Regression is a good interpretable baseline; compare with Random Forest.")
    if has_regression:
        if n_rows >= 500:
            recommendation_lines.append("Regression: Random Forest Regressor is a strong default baseline for non-linear relationships.")
        else:
            recommendation_lines.append("Regression: Linear Regression is a fast baseline; compare with Random Forest Regressor.")
    if has_numeric and analysis["rows"] >= 20 and len(analysis["numeric_cols"]) >= 2:
        recommendation_lines.append("Clustering: KMeans is suitable for segmentation after scaling numeric features.")
    if not recommendation_lines:
        recommendation_lines.append("No clear training recommendation yet. Add more rows/features or define a target column.")

    if has_classification and has_regression:
        best_model = "Best starting family depends on target type: Random Forest (Classifier/Regressor) is the most robust first choice."
    elif has_classification:
        best_model = "Best starting model: Random Forest Classifier."
    elif has_regression:
        best_model = "Best starting model: Random Forest Regressor."
    elif has_numeric and analysis["rows"] >= 20 and len(analysis["numeric_cols"]) >= 2:
        best_model = "Best starting model: KMeans (unsupervised clustering)."
    else:
        best_model = "Best model cannot be determined until a suitable target or more numeric data is available."

    return {
        "recommendation_lines": recommendation_lines,
        "best_model": best_model,
    }


def suggest_predictor_columns(
    df: pd.DataFrame,
    target_col: str,
    candidate_cols: List[str],
    problem_type: str,
    max_features: int = 15,
) -> List[str]:
    """Auto-select predictor columns based on data quality and target relevance."""
    if target_col not in df.columns:
        return []

    valid_cols = []
    for col in candidate_cols:
        if col == target_col or col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        missing_ratio = float(df[col].isna().mean()) if len(df) else 0.0
        if missing_ratio > 0.6:
            continue
        valid_cols.append(col)

    if not valid_cols:
        return []

    target_non_null = df[target_col].dropna()
    if target_non_null.empty:
        return valid_cols[: min(max_features, len(valid_cols))]

    ranked = []
    if problem_type == "regression" and pd.api.types.is_numeric_dtype(df[target_col]):
        for col in valid_cols:
            try:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                score = abs(df[[col, target_col]].dropna().corr().iloc[0, 1])
                if not np.isnan(score):
                    ranked.append((col, float(score)))
            except Exception:
                continue
    elif problem_type == "classification":
        y_codes = pd.Series(pd.factorize(df[target_col])[0], index=df.index)
        for col in valid_cols:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    score = abs(pd.concat([df[col], y_codes], axis=1).dropna().corr().iloc[0, 1])
                    if np.isnan(score):
                        score = 0.0
                    ranked.append((col, float(score)))
                else:
                    # Prefer lower-cardinality categorical features for stability.
                    cardinality_penalty = 1 / (1 + df[col].nunique(dropna=True))
                    ranked.append((col, float(cardinality_penalty)))
            except Exception:
                continue

    if ranked:
        ranked_sorted = [name for name, _ in sorted(ranked, key=lambda item: item[1], reverse=True)]
        selected = ranked_sorted[: min(max_features, len(ranked_sorted))]
    else:
        selected = valid_cols[: min(max_features, len(valid_cols))]

    return selected


def _prepare_model_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Prepare model features and target for scikit-learn training."""
    if target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float), f"Target column '{target_col}' was not found."

    if not feature_cols:
        return pd.DataFrame(), pd.Series(dtype=float), "Select at least one predictor column."

    if target_col in feature_cols:
        return pd.DataFrame(), pd.Series(dtype=float), "Target column cannot be used as a predictor."

    data = df.copy()
    y = data[target_col].dropna()

    if y.empty:
        return pd.DataFrame(), pd.Series(dtype=float), "Target column has no valid values after removing missing data."

    missing_features = [col for col in feature_cols if col not in data.columns]
    if missing_features:
        return pd.DataFrame(), pd.Series(dtype=float), f"Predictor column(s) not found: {', '.join(missing_features)}."

    X = data.loc[y.index, feature_cols].copy()
    if X.empty:
        return pd.DataFrame(), pd.Series(dtype=float), "No feature columns are available for training."

    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(y):
        return pd.DataFrame(), pd.Series(dtype=float), "Regression target must be numeric."

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if numeric_cols:
        X[numeric_cols] = X[numeric_cols].apply(lambda series: series.fillna(series.median()))
    if categorical_cols:
        for col in categorical_cols:
            mode_value = X[col].mode(dropna=True)
            if mode_value.empty:
                X[col] = X[col].fillna("missing")
            else:
                X[col] = X[col].fillna(mode_value[0]).astype(str)

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y, ""


def _prepare_classification_target(y: pd.Series) -> Tuple[pd.Series, str]:
    """Ensure classification target is discrete and compatible with sklearn classifiers."""
    y = y.dropna()
    if y.empty:
        return pd.Series(dtype=float), "Target column has no valid values after removing missing data."

    unique_count = int(y.nunique(dropna=True))
    if unique_count < 2:
        return pd.Series(dtype=float), "Classification target must contain at least two classes."

    # If the target looks highly unique relative to rows, it is likely continuous.
    unique_ratio = unique_count / len(y) if len(y) else 1.0
    if unique_count > 30 and unique_ratio > 0.2:
        return (
            pd.Series(dtype=float),
            "Selected target appears continuous for classification. Choose a categorical/discrete target or use Regression.",
        )

    if pd.api.types.is_numeric_dtype(y):
        y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().any():
            return pd.Series(dtype=float), "Target contains invalid numeric values."

        # Integer-like numeric targets are common class labels (0/1, 1/2/3, etc.).
        if np.all(np.isclose(y_numeric, np.round(y_numeric), atol=1e-9)):
            return pd.Series(np.round(y_numeric).astype(int), index=y.index), ""

        # Non-integer numeric values can still represent classes if low-cardinality.
        if unique_count <= 20:
            return y.astype(str), ""

        return (
            pd.Series(dtype=float),
            "Selected target appears continuous for classification. Choose a categorical/discrete target or use Regression.",
        )

    # Treat non-numeric labels as categorical classes.
    return y.astype(str), ""


def train_classification_model(
    df: pd.DataFrame,
    model_name: str,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    model_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train a classification model and return metrics."""
    try:
        model_params = model_params or {}
        X, y, error = _prepare_model_data(df, target_col, feature_cols, "classification")
        if error:
            return {"error": error}

        y, target_error = _prepare_classification_target(y)
        if target_error:
            return {"error": target_error}

        if y.nunique() < 2:
            return {"error": "Classification target must contain at least two classes."}

        stratify_target = y if y.value_counts().min() >= 2 else None
        if len(X) < 2:
            return {"error": "Not enough rows to train a classification model."}

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_target,
        )
        
        if model_name == "Logistic Regression":
            model = LogisticRegression(
                penalty=model_params.get("penalty", "l2"),
                C=model_params.get("C", 1.0),
                solver=model_params.get("solver", "lbfgs"),
                max_iter=model_params.get("max_iter", 1000),
                fit_intercept=model_params.get("fit_intercept", True),
                class_weight=model_params.get("class_weight", None),
                l1_ratio=model_params.get("l1_ratio", None),
                random_state=42,
            )
        elif model_name == "KNN":
            model = KNeighborsClassifier(
                n_neighbors=model_params.get("n_neighbors", 5),
                weights=model_params.get("weights", "uniform"),
                algorithm=model_params.get("algorithm", "auto"),
                leaf_size=model_params.get("leaf_size", 30),
                p=model_params.get("p", 2),
                metric=model_params.get("metric", "minkowski"),
            )
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=model_params.get("criterion", "gini"),
                splitter=model_params.get("splitter", "best"),
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                min_weight_fraction_leaf=model_params.get("min_weight_fraction_leaf", 0.0),
                max_features=model_params.get("max_features", None),
                max_leaf_nodes=model_params.get("max_leaf_nodes", None),
                min_impurity_decrease=model_params.get("min_impurity_decrease", 0.0),
                class_weight=model_params.get("class_weight", None),
                ccp_alpha=model_params.get("ccp_alpha", 0.0),
                random_state=42,
            )
        elif model_name == "Random Forest Classifier":
            model = RandomForestClassifier(
                n_estimators=model_params.get("n_estimators", 100),
                criterion=model_params.get("criterion", "gini"),
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                min_weight_fraction_leaf=model_params.get("min_weight_fraction_leaf", 0.0),
                max_features=model_params.get("max_features", "sqrt"),
                max_leaf_nodes=model_params.get("max_leaf_nodes", None),
                min_impurity_decrease=model_params.get("min_impurity_decrease", 0.0),
                bootstrap=model_params.get("bootstrap", True),
                oob_score=model_params.get("oob_score", False),
                n_jobs=model_params.get("n_jobs", -1),
                class_weight=model_params.get("class_weight", None),
                ccp_alpha=model_params.get("ccp_alpha", 0.0),
                max_samples=model_params.get("max_samples", None),
                random_state=42,
            )
        elif model_name == "SVM":
            model = SVC(
                C=model_params.get("C", 1.0),
                kernel=model_params.get("kernel", "rbf"),
                degree=model_params.get("degree", 3),
                gamma=model_params.get("gamma", "scale"),
                coef0=model_params.get("coef0", 0.0),
                shrinking=model_params.get("shrinking", True),
                probability=model_params.get("probability", False),
                tol=model_params.get("tol", 1e-3),
                class_weight=model_params.get("class_weight", None),
                max_iter=model_params.get("max_iter", -1),
                decision_function_shape=model_params.get("decision_function_shape", "ovr"),
                break_ties=model_params.get("break_ties", False),
                random_state=42,
            )
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
            "feature_names": X.columns.tolist(),
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
            "X_train_df": X_train,
            "y_train_series": y_train,
        }
    except Exception as e:
        return {"error": str(e)}


def train_regression_model(
    df: pd.DataFrame,
    model_name: str,
    target_col: str,
    feature_cols: List[str],
    test_size: float = 0.2,
    model_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train a regression model and return metrics."""
    try:
        model_params = model_params or {}
        X, y, error = _prepare_model_data(df, target_col, feature_cols, "regression")
        if error:
            return {"error": error}

        if len(X) < 2:
            return {"error": "Not enough rows to train a regression model."}

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
        )
        
        if model_name == "Linear Regression":
            model = LinearRegression(
                fit_intercept=model_params.get("fit_intercept", True),
                copy_X=model_params.get("copy_X", True),
                n_jobs=model_params.get("n_jobs", None),
                positive=model_params.get("positive", False),
            )
        elif model_name == "KNN":
            model = KNeighborsRegressor(
                n_neighbors=model_params.get("n_neighbors", 5),
                weights=model_params.get("weights", "uniform"),
                algorithm=model_params.get("algorithm", "auto"),
                leaf_size=model_params.get("leaf_size", 30),
                p=model_params.get("p", 2),
                metric=model_params.get("metric", "minkowski"),
            )
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(
                criterion=model_params.get("criterion", "squared_error"),
                splitter=model_params.get("splitter", "best"),
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                min_weight_fraction_leaf=model_params.get("min_weight_fraction_leaf", 0.0),
                max_features=model_params.get("max_features", None),
                max_leaf_nodes=model_params.get("max_leaf_nodes", None),
                min_impurity_decrease=model_params.get("min_impurity_decrease", 0.0),
                ccp_alpha=model_params.get("ccp_alpha", 0.0),
                random_state=42,
            )
        elif model_name == "Random Forest Regressor":
            model = RandomForestRegressor(
                n_estimators=model_params.get("n_estimators", 100),
                criterion=model_params.get("criterion", "squared_error"),
                max_depth=model_params.get("max_depth", None),
                min_samples_split=model_params.get("min_samples_split", 2),
                min_samples_leaf=model_params.get("min_samples_leaf", 1),
                min_weight_fraction_leaf=model_params.get("min_weight_fraction_leaf", 0.0),
                max_features=model_params.get("max_features", 1.0),
                max_leaf_nodes=model_params.get("max_leaf_nodes", None),
                min_impurity_decrease=model_params.get("min_impurity_decrease", 0.0),
                bootstrap=model_params.get("bootstrap", True),
                oob_score=model_params.get("oob_score", False),
                n_jobs=model_params.get("n_jobs", -1),
                ccp_alpha=model_params.get("ccp_alpha", 0.0),
                max_samples=model_params.get("max_samples", None),
                random_state=42,
            )
        elif model_name == "SVM":
            model = SVR(
                C=model_params.get("C", 1.0),
                kernel=model_params.get("kernel", "rbf"),
                degree=model_params.get("degree", 3),
                gamma=model_params.get("gamma", "scale"),
                coef0=model_params.get("coef0", 0.0),
                tol=model_params.get("tol", 1e-3),
                epsilon=model_params.get("epsilon", 0.1),
                shrinking=model_params.get("shrinking", True),
                cache_size=model_params.get("cache_size", 200.0),
                max_iter=model_params.get("max_iter", -1),
            )
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
            "feature_names": X.columns.tolist(),
            "y_test": y_test.reset_index(drop=True),
            "y_pred": pd.Series(y_pred),
            "X_test_vals": X_test.iloc[:, 0].values.tolist() if len(X.columns) == 1 else [],
            "X_train_df": X_train,
            "y_train_series": y_train,
        }
    except Exception as e:
        return {"error": str(e)}


def train_clustering_model(
    df: pd.DataFrame,
    n_clusters: int,
    numeric_cols: List[str],
    model_params: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Train a KMeans clustering model and return metrics."""
    try:
        model_params = model_params or {}
        if not numeric_cols:
            return {"error": "No numeric columns selected."}

        X = df[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        X = X.dropna()

        if len(X) == 0:
            return {"error": "No valid data for clustering."}
        
        if n_clusters > len(X):
            return {"error": "Number of clusters cannot exceed the number of available samples."}
        
        model = KMeans(
            n_clusters=n_clusters,
            init=model_params.get("init", "k-means++"),
            n_init=model_params.get("n_init", 10),
            max_iter=model_params.get("max_iter", 300),
            tol=model_params.get("tol", 1e-4),
            algorithm=model_params.get("algorithm", "lloyd"),
            random_state=42,
        )
        clusters = model.fit_predict(X)
        inertia = model.inertia_
        
        return {
            "model": model,
            "clusters": clusters,
            "inertia": inertia,
            "n_samples": len(X),
            "n_clusters": n_clusters,
            "cluster_data": X.reset_index(drop=True),
            "feature_names": numeric_cols,
            "cluster_centers": model.cluster_centers_,
        }
    except Exception as e:
        return {"error": str(e)}


def plot_classification_visualizations(results: Dict[str, Any], model_name: str) -> None:
    y_test_raw = results.get("y_test", [])
    y_pred_raw = results.get("y_pred", [])
    y_test = pd.Series(np.array(y_test_raw).ravel()).astype(str)
    y_pred = pd.Series(np.array(y_pred_raw).ravel()).astype(str)
    feature_names = results.get("feature_names", [])
    model = results.get("model")

    if y_test.empty or y_pred.empty:
        st.info("No prediction data available for classification visualizations.")
        return

    st.subheader(f"Automatic Visualizations: {model_name}")

    tabs_list = ["Confusion Matrix", "Feature Influence"]
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        tabs_list.append("Decision Tree Diagram")
    elif isinstance(model, (KNeighborsClassifier, SVC)):
        tabs_list.append("Decision Boundary Map")
        
    tabs = st.tabs(tabs_list)

    with tabs[0]:
        labels = sorted(set(y_test.tolist() + y_pred.tolist()))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        render_figure(fig)

    with tabs[1]:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importance.values, y=importance.index, orient="h", hue=importance.index, legend=False, palette="viridis", ax=ax)
            ax.set_title("Top Feature Importances")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            render_figure(fig)
        elif hasattr(model, "coef_"):
            coef_values = model.coef_
            if isinstance(coef_values, np.ndarray) and coef_values.ndim > 1:
                coef_values = np.mean(np.abs(coef_values), axis=0)
            coef_series = pd.Series(np.asarray(coef_values).flatten(), index=feature_names)
            coef_series = coef_series.sort_values(key=np.abs, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=coef_series.values, y=coef_series.index, orient="h", hue=coef_series.index, legend=False, palette="magma", ax=ax)
            ax.set_title("Top Coefficients")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            render_figure(fig)
        else:
            st.info("This model does not expose feature influence directly.")

    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 8))
            tree_to_plot = model.estimators_[0] if isinstance(model, RandomForestClassifier) else model
            plot_tree(tree_to_plot, feature_names=feature_names, class_names=True, filled=True, ax=ax, max_depth=3)
            title = "Random Forest - First Tree Diagram (max depth 3)" if isinstance(model, RandomForestClassifier) else "Decision Tree Diagram (max depth 3)"
            ax.set_title(title)
            render_figure(fig)
    elif isinstance(model, (KNeighborsClassifier, SVC)):
        with tabs[2]:
            try:
                X_train_df = results.get("X_train_df")
                y_train_series = results.get("y_train_series")
                
                if X_train_df is not None and y_train_series is not None:
                    X_train = getattr(X_train_df, "values", X_train_df)
                    y_train = getattr(y_train_series, "values", y_train_series)
                    y_train_codes, _ = pd.factorize(pd.Series(y_train).astype(str))
                    
                    if X_train.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X_vis = pca.fit_transform(X_train)
                        vis_names = ["PCA Component 1", "PCA Component 2"]
                        
                        import copy
                        from sklearn.base import clone
                        vis_model = clone(model)
                        vis_model.fit(X_vis, y_train)
                    else:
                        X_vis = X_train
                        vis_names = feature_names if len(feature_names) == 2 else ["Feature 1", "Feature 2"]
                        vis_model = model
                        
                    fig, ax = plt.subplots(figsize=(8, 6))
                    DecisionBoundaryDisplay.from_estimator(vis_model, X_vis, response_method="predict", alpha=0.4, ax=ax, cmap="coolwarm")
                    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train_codes, edgecolors='k', cmap="coolwarm")
                    
                    model_title_name = "KNN" if isinstance(model, KNeighborsClassifier) else "SVM"
                    ax.set_title(f"{model_title_name} Decision Boundary")
                    ax.set_xlabel(vis_names[0])
                    ax.set_ylabel(vis_names[1])
                    render_figure(fig)
                else:
                    st.info("Training data not available in results dict to plot decision boundary.")
            except Exception as e:
                st.info(f"Could not generate Decision Boundary: {e}")


def plot_regression_visualizations(results: Dict[str, Any], model_name: str) -> None:
    y_test_raw = results.get("y_test", [])
    y_pred_raw = results.get("y_pred", [])
    y_test = pd.Series(np.array(y_test_raw).ravel()).astype(float)
    y_pred = pd.Series(np.array(y_pred_raw).ravel()).astype(float)
    feature_names = results.get("feature_names", [])
    model = results.get("model")

    if y_test.empty or y_pred.empty:
        st.info("No prediction data available for regression visualizations.")
        return

    st.subheader(f"Automatic Visualizations: {model_name}")

    tabs_list = ["Residuals", "Feature Influence"]
    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        tabs_list.append("Decision Tree Diagram")
    elif isinstance(model, (KNeighborsRegressor, SVR)):
        tabs_list.append("Regression Value Map")
        
    tabs = st.tabs(tabs_list)

    with tabs[0]:
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, color="#d9683f", ax=ax)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual (Actual - Predicted)")
        render_figure(fig)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, ax=ax2)
        ax2.axhline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_title("Residuals vs Predicted")
        ax2.set_xlabel("Predicted Values")
        ax2.set_ylabel("Residual")
        render_figure(fig2)

    with tabs[1]:
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=importance.values, y=importance.index, orient="h", hue=importance.index, legend=False, palette="viridis", ax=ax)
            ax.set_title("Top Feature Importances")
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            render_figure(fig)
        elif hasattr(model, "coef_"):
            coef_series = pd.Series(np.asarray(model.coef_).flatten(), index=feature_names)
            coef_series = coef_series.sort_values(key=np.abs, ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=coef_series.values, y=coef_series.index, orient="h", hue=coef_series.index, legend=False, palette="magma", ax=ax)
            ax.set_title("Top Coefficients")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            render_figure(fig)
        else:
            st.info("This model does not expose feature influence directly.")

    if isinstance(model, (DecisionTreeRegressor, RandomForestRegressor)):
        with tabs[2]:
            fig, ax = plt.subplots(figsize=(12, 8))
            tree_to_plot = model.estimators_[0] if isinstance(model, RandomForestRegressor) else model
            plot_tree(tree_to_plot, feature_names=feature_names, filled=True, ax=ax, max_depth=3)
            title = "Random Forest - First Tree Diagram (max depth 3)" if isinstance(model, RandomForestRegressor) else "Decision Tree Diagram (max depth 3)"
            ax.set_title(title)
            render_figure(fig)
    elif isinstance(model, (KNeighborsRegressor, SVR)):
        with tabs[2]:
            try:
                X_train_df = results.get("X_train_df")
                y_train_series = results.get("y_train_series")
                
                if X_train_df is not None and y_train_series is not None:
                    X_train = getattr(X_train_df, "values", X_train_df)
                    y_train = getattr(y_train_series, "values", y_train_series)
                    
                    if X_train.shape[1] > 2:
                        pca = PCA(n_components=2)
                        X_vis = pca.fit_transform(X_train)
                        vis_names = ["PCA Component 1", "PCA Component 2"]
                        
                        from sklearn.base import clone
                        vis_model = clone(model)
                        vis_model.fit(X_vis, y_train)
                    elif X_train.shape[1] == 1:
                        st.info("Regression value map is not available for single-feature training data.")
                        vis_model = None
                    else:
                        X_vis = X_train
                        vis_names = feature_names if len(feature_names) == 2 else ["Feature 1", "Feature 2"]
                        vis_model = model
                        
                    if vis_model:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        DecisionBoundaryDisplay.from_estimator(vis_model, X_vis, response_method="predict", alpha=0.6, ax=ax, cmap="coolwarm")
                        scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, edgecolors='k', cmap="coolwarm")
                        
                        model_title_name = "KNN" if isinstance(model, KNeighborsRegressor) else "SVM"
                        ax.set_title(f"{model_title_name} Regression Value Map")
                        ax.set_xlabel(vis_names[0])
                        ax.set_ylabel(vis_names[1])
                        fig.colorbar(scatter, ax=ax, label="Target Predict Map")
                        render_figure(fig)
                else:
                    st.info("Training data not available in results dict to plot Regression Map.")
            except Exception as e:
                st.info(f"Could not generate Regression Map: {e}")


def plot_clustering_visualizations(results: Dict[str, Any], selected_cols: List[str]) -> None:
    clusters = np.asarray(results.get("clusters", []))
    cluster_data = results.get("cluster_data", pd.DataFrame())
    centers = np.asarray(results.get("cluster_centers", []))

    if cluster_data is None or len(clusters) == 0:
        st.info("No cluster data available for visualizations.")
        return

    st.subheader("Automatic Visualizations: KMeans")

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Cluster Scatter", "Cluster Sizes", "Centroid Heatmap"])

    with viz_tab1:
        if len(selected_cols) >= 2 and not cluster_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                cluster_data.iloc[:, 0],
                cluster_data.iloc[:, 1],
                c=clusters,
                cmap="viridis",
                alpha=0.7,
            )
            ax.set_xlabel(selected_cols[0])
            ax.set_ylabel(selected_cols[1])
            ax.set_title("Cluster Scatter (First Two Selected Features)")
            plt.colorbar(scatter, ax=ax, label="Cluster")
            render_figure(fig)
        else:
            st.info("Select at least two columns to display cluster scatter plot.")

    with viz_tab2:
        counts = pd.Series(clusters).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.index.astype(str), y=counts.values, hue=counts.index.astype(str), legend=False, palette="Set2", ax=ax)
        ax.set_title("Samples per Cluster")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")
        render_figure(fig)

    with viz_tab3:
        if centers.size > 0:
            centers_df = pd.DataFrame(centers, columns=selected_cols)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
            ax.set_title("Cluster Centroids by Feature")
            ax.set_xlabel("Feature")
            ax.set_ylabel("Cluster")
            render_figure(fig)
        else:
            st.info("Centroid data is not available for this run.")


def main():
    apply_ui_theme()

    st.markdown(
        """
        <div class="hero-shell">
            <h1 class="hero-title">Automated EDA and Data Preprocessing Studio</h1>
            <p class="hero-subtitle">
                Upload any CSV dataset to profile data quality, clean and transform features, run exploratory analysis,
                and train machine learning models from one streamlined interface.
            </p>
            <div class="hero-meta">
                <span class="hero-chip">Data Cleaning</span>
                <span class="hero-chip">EDA Visuals</span>
                <span class="hero-chip">ML Training</span>
                <span class="hero-chip">Export Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sidebar = st.sidebar
    sidebar.markdown("<h3 class='sidebar-label'>Workspace</h3>", unsafe_allow_html=True)
    uploaded_file = sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to begin automated exploratory data analysis and preprocessing.")
        st.stop()

    sidebar.markdown(
        """
        <div class="sidebar-nav-wrap">
            <h4 class="sidebar-nav-title">Process Navigator</h4>
            <div class="sidebar-nav-list">
                <a class="sidebar-nav-link" href="#dataset-overview">Dataset Overview</a>
                <a class="sidebar-nav-link" href="#automated-dataset-intelligence">Automated Dataset Intelligence</a>
                <a class="sidebar-nav-link" href="#missing-values-before-cleaning">Missing Values Before Cleaning</a>
                <a class="sidebar-nav-link" href="#data-cleaning">Data Cleaning</a>
                <a class="sidebar-nav-link" href="#encoding">Encoding</a>
                <a class="sidebar-nav-link" href="#feature-engineering">Feature Engineering</a>
                <a class="sidebar-nav-link" href="#exploratory-data-analysis">Exploratory Data Analysis</a>
                <a class="sidebar-nav-link" href="#scaling-and-pca">Scaling and PCA</a>
                <a class="sidebar-nav-link" href="#outlier-detection">Outlier Detection</a>
                <a class="sidebar-nav-link" href="#model-training">Model Training</a>
                <a class="sidebar-nav-link" href="#pipeline-summary">Pipeline Summary</a>
                <a class="sidebar-nav-link" href="#notes">Notes</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Loading dataset..."):
            original_df = load_data(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read this file as CSV. {exc}")
        st.stop()

    if "df" not in st.session_state or st.session_state.get("uploaded_file_name") != uploaded_file.name:
        st.session_state.df = normalize_column_names(original_df.copy())
        st.session_state.uploaded_file_name = uploaded_file.name
    df = st.session_state.df
    pipeline_steps = ["Normalized column names"]

    if df.shape[1] == 0:
        st.error("The uploaded dataset has no columns after parsing. Upload a valid tabular CSV file.")
        st.stop()

    quick1, quick2, quick3, quick4 = st.columns(4)
    quick1.metric("Dataset", uploaded_file.name)
    quick2.metric("Rows", len(df))
    quick3.metric("Columns", len(df.columns))
    quick4.metric("Missing Cells", int(df.isna().sum().sum()))

    st.markdown("<div id='dataset-overview'></div>", unsafe_allow_html=True)
    render_section_header("Dataset Overview")
    st.write("**Preview of the uploaded dataset**")
    st.dataframe(df.head(10))
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))
    st.write("**Data types:**")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}))

    st.markdown("---")
    st.markdown("<div id='automated-dataset-intelligence'></div>", unsafe_allow_html=True)
    render_section_header("Automated Dataset Intelligence")
    analysis = analyze_dataset(df)
    suggestions = preprocessing_suggestions(analysis)
    model_advice = model_recommendations(analysis)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", analysis["rows"])
    m2.metric("Columns", analysis["columns"])
    m3.metric("Missing %", f"{analysis['missing_percent_total']}%")
    m4.metric("Duplicate Rows", analysis["duplicate_rows"])
    m5.metric("Numeric Columns", len(analysis["numeric_cols"]))

    st.subheader("Detailed Column Profile")
    st.dataframe(analysis["column_profile"], use_container_width=True)

    st.subheader("Suggested Preprocessing Plan")
    for idx, suggestion in enumerate(suggestions, start=1):
        st.write(f"{idx}. {suggestion}")

    st.subheader("Model Recommendation")
    st.info(model_advice["best_model"])
    for idx, recommendation in enumerate(model_advice["recommendation_lines"], start=1):
        st.write(f"{idx}. {recommendation}")

    if analysis["classification_targets"]:
        st.write("Classification target candidates:", analysis["classification_targets"][:10])
    if analysis["regression_targets"]:
        st.write("Regression target candidates:", analysis["regression_targets"][:10])

    st.markdown("---")
    st.markdown("<div id='missing-values-before-cleaning'></div>", unsafe_allow_html=True)
    render_section_header("Missing Values Before Cleaning")
    original_missing_summary = get_missing_summary(df)
    if original_missing_summary["missing_count"].sum() == 0:
        st.success("No missing values detected in the uploaded dataset.")
    else:
        st.write("Review missing values before applying cleaning steps.")
        st.dataframe(original_missing_summary)

    st.markdown("---")
    st.markdown("<div id='data-cleaning'></div>", unsafe_allow_html=True)
    render_section_header("Data Cleaning")
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
    st.markdown("<div id='encoding'></div>", unsafe_allow_html=True)
    render_section_header("Encoding")
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
    st.markdown("<div id='feature-engineering'></div>", unsafe_allow_html=True)
    render_section_header("Feature Engineering")
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
            if operation in ["-", "*", "/"] and not (
                pd.api.types.is_numeric_dtype(df[left_column]) and pd.api.types.is_numeric_dtype(df[right_column])
            ):
                st.warning("For -, * and / operations, both selected columns must be numeric.")
            else:
                with st.spinner("Creating new feature..."):
                    df = create_combined_feature(df, new_feature_name, left_column, right_column, operation)
                st.session_state.df = df
                pipeline_steps.append(f"Created feature {new_feature_name} = {left_column} {operation} {right_column}")
                st.success(f"Feature '{new_feature_name}' created.")

    st.markdown("---")
    st.markdown("<div id='exploratory-data-analysis'></div>", unsafe_allow_html=True)
    render_section_header("Exploratory Data Analysis")
    if st.checkbox("Show summary statistics", value=True):
        with st.spinner("Computing summary statistics..."):
            st.dataframe(generate_summary_statistics(df))

    if st.checkbox("Show top correlated feature pairs", value=True):
        with st.spinner("Calculating top correlations..."):
            show_top_correlations(df, top_n=5)

    if st.checkbox("Show correlation heatmap", value=True):
        with st.spinner("Rendering correlation heatmap..."):
            plot_correlation_heatmap(df)

    if numeric_columns:
        dist_column = st.selectbox("Select numeric column for distribution plot", options=numeric_columns, index=0)
        plot_type = st.selectbox("Distribution plot type", options=["Histogram", "KDE"])
        if st.button("Show distribution plot"):
            plot_distribution(df, dist_column, plot_type)
    else:
        st.info("No numeric columns available for distribution plots.")

    if categorical_columns:
        cat_col_for_count = st.selectbox("Select categorical column for count plot", options=categorical_columns)
        if st.button("Show count plot"):
            plot_countplot(df, cat_col_for_count)

    st.markdown("---")
    st.markdown("<div id='scaling-and-pca'></div>", unsafe_allow_html=True)
    render_section_header("Scaling and PCA")

    st.subheader("Scaling")
    scaling_method = st.selectbox(
        "Scaling method for numeric columns",
        options=["None", "StandardScaler", "MinMaxScaler"],
        index=0,
    )

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

    st.subheader("PCA")
    st.caption("PCA is independent from scaling. You can run either step separately.")
    selected_pca_cols = st.multiselect(
        "Select numeric columns for PCA",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
        help="PCA will be computed only on these selected numeric columns.",
        key="pca_selected_columns",
    )

    pca_component_limit = min(len(selected_pca_cols), len(processed_df))
    pca_component_options = [count for count in [2, 3] if count <= pca_component_limit]
    if pca_component_options:
        default_index = 0
        n_components = st.selectbox(
            "Number of PCA components",
            options=pca_component_options,
            index=default_index,
            help="Select 2 for a 2D plot or 3 for a 3D plot.",
        )
    else:
        n_components = 2
        st.info("PCA needs at least 2 selected numeric columns and 2 rows.")

    if st.button("Run PCA"):
        if not selected_pca_cols:
            st.warning("Select at least two numeric columns to run PCA.")
        elif not pca_component_options:
            st.warning("PCA needs at least 2 selected numeric columns and 2 rows.")
        else:
            with st.spinner("Running PCA..."):
                pca_df, variance, pca_error = apply_pca(processed_df, n_components, selected_pca_cols)
            if pca_error:
                st.warning(pca_error)
            else:
                st.subheader("PCA Results")
                st.write("Selected PCA columns:", selected_pca_cols)
                st.write("Explained variance ratio:")
                st.write(variance.round(4))
                plot_pca_projection(pca_df, variance)
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
    st.markdown("<div id='outlier-detection'></div>", unsafe_allow_html=True)
    render_section_header("Outlier Detection")
    st.write("Identify potential outliers in numeric columns using the IQR rule (values outside 1.5 x IQR).")

    outlier_summary, outlier_mask = summarize_outliers_iqr(processed_df)
    if outlier_summary.empty:
        st.success("No potential outliers were detected in numeric columns using the IQR rule.")
    else:
        total_outlier_rows = int(outlier_mask.sum())
        st.metric("Rows With At Least One Outlier", total_outlier_rows)
        st.dataframe(outlier_summary, use_container_width=True)

        with st.expander("Preview rows with outliers"):
            preview_limit = st.slider(
                "Rows to preview",
                min_value=5,
                max_value=min(200, max(5, total_outlier_rows)),
                value=min(20, max(5, total_outlier_rows)),
                key="outlier_preview_limit",
            )
            st.dataframe(processed_df[outlier_mask].head(preview_limit), use_container_width=True)

    st.markdown("---")
    st.markdown("<div id='model-training'></div>", unsafe_allow_html=True)
    render_section_header("Model Training")
    
    model_type = st.selectbox(
        "Select Model Type",
        options=["Classification", "Regression", "Clustering"],
        index=0,
    )
    
    if model_type == "Classification":
        st.subheader("Classification Models")
        st.write("Train a classification model to predict categorical targets.")
        
        all_cols = processed_df.columns.tolist()

        if len(all_cols) < 2 or len(processed_df) < 2:
            st.warning("Classification requires at least 2 columns and 2 rows.")
        else:
            target_col = st.selectbox(
                "Select target column (must be numeric or encoded categorical)",
                options=all_cols,
                key="class_target",
            )

            predictor_options = [col for col in all_cols if col != target_col]
            auto_class_features = suggest_predictor_columns(
                processed_df,
                target_col,
                predictor_options,
                problem_type="classification",
            )
            if st.session_state.get("class_target_prev") != target_col:
                st.session_state["class_features"] = auto_class_features
            st.session_state["class_target_prev"] = target_col
            feature_cols = st.multiselect(
                "Select predictor columns (auto-selected)",
                options=predictor_options,
                default=auto_class_features,
                key="class_features",
                help="Choose the input columns used to predict the target column.",
            )
            
            model_name = st.selectbox(
                "Select Classification Model",
                options=["Logistic Regression", "KNN", "SVM", "Decision Tree", "Random Forest Classifier"],
                key="class_model",
            )

            model_params = {}
            st.write("Model Hyperparameters")

            if model_name == "Logistic Regression":
                class_penalty = st.selectbox("penalty", ["l2", "l1", "elasticnet", "none"], index=0, key="class_lr_penalty")
                model_params["penalty"] = None if class_penalty == "none" else class_penalty
                model_params["C"] = st.number_input("C", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1, key="class_lr_c")
                model_params["solver"] = st.selectbox(
                    "solver",
                    ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
                    index=0,
                    key="class_lr_solver",
                )
                model_params["max_iter"] = st.number_input("max_iter", min_value=100, max_value=10000, value=1000, step=100, key="class_lr_max_iter")
                model_params["fit_intercept"] = st.checkbox("fit_intercept", value=True, key="class_lr_fit_intercept")
                class_weight_opt = st.selectbox("class_weight", ["None", "balanced"], index=0, key="class_lr_class_weight")
                model_params["class_weight"] = None if class_weight_opt == "None" else class_weight_opt
                model_params["l1_ratio"] = st.number_input("l1_ratio (used for elasticnet)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="class_lr_l1_ratio")

            elif model_name == "KNN":
                model_params["n_neighbors"] = st.number_input("n_neighbors", min_value=1, max_value=100, value=5, step=1, key="class_knn_neighbors")
                model_params["weights"] = st.selectbox("weights", ["uniform", "distance"], index=0, key="class_knn_weights")
                model_params["algorithm"] = st.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0, key="class_knn_algorithm")
                model_params["leaf_size"] = st.number_input("leaf_size", min_value=1, max_value=200, value=30, step=1, key="class_knn_leaf_size")
                model_params["p"] = st.number_input("p", min_value=1, max_value=10, value=2, step=1, key="class_knn_p")
                model_params["metric"] = st.selectbox("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"], index=0, key="class_knn_metric")

            elif model_name == "SVM":
                model_params["C"] = st.number_input("C", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1, key="class_svm_c")
                model_params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"], index=1, key="class_svm_kernel")
                if model_params["kernel"] == "poly":
                    model_params["degree"] = st.number_input("degree", min_value=1, max_value=10, value=3, step=1, key="class_svm_degree")
                else:
                    model_params["degree"] = 3
                model_params["gamma"] = st.selectbox("gamma", ["scale", "auto"], index=0, key="class_svm_gamma")
                model_params["coef0"] = st.number_input("coef0", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="class_svm_coef0")
                model_params["shrinking"] = st.checkbox("shrinking", value=True, key="class_svm_shrinking")
                model_params["probability"] = st.checkbox("probability", value=False, key="class_svm_probability")
                model_params["tol"] = st.number_input("tol", min_value=0.000001, max_value=1.0, value=0.001, step=0.0001, format="%.6f", key="class_svm_tol")
                class_weight_opt = st.selectbox("class_weight", ["None", "balanced"], index=0, key="class_svm_class_weight")
                model_params["class_weight"] = None if class_weight_opt == "None" else class_weight_opt
                model_params["max_iter"] = st.number_input("max_iter (-1 for no limit)", min_value=-1, max_value=20000, value=-1, step=1, key="class_svm_max_iter")
                model_params["decision_function_shape"] = st.selectbox("decision_function_shape", ["ovr", "ovo"], index=0, key="class_svm_dfs")
                model_params["break_ties"] = st.checkbox("break_ties", value=False, key="class_svm_break_ties")

            elif model_name == "Decision Tree":
                model_params["criterion"] = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0, key="class_dt_criterion")
                model_params["splitter"] = st.selectbox("splitter", ["best", "random"], index=0, key="class_dt_splitter")
                class_dt_max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=200, value=0, step=1, key="class_dt_max_depth")
                model_params["max_depth"] = None if class_dt_max_depth == 0 else int(class_dt_max_depth)
                model_params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=100, value=2, step=1, key="class_dt_min_samples_split")
                model_params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="class_dt_min_samples_leaf")
                model_params["min_weight_fraction_leaf"] = st.number_input("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="class_dt_min_weight_fraction_leaf")
                class_dt_max_features = st.selectbox("max_features", ["None", "sqrt", "log2"], index=0, key="class_dt_max_features")
                model_params["max_features"] = None if class_dt_max_features == "None" else class_dt_max_features
                class_dt_max_leaf_nodes = st.number_input("max_leaf_nodes (0 = None)", min_value=0, max_value=2000, value=0, step=1, key="class_dt_max_leaf_nodes")
                model_params["max_leaf_nodes"] = None if class_dt_max_leaf_nodes == 0 else int(class_dt_max_leaf_nodes)
                model_params["min_impurity_decrease"] = st.number_input("min_impurity_decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="class_dt_min_impurity_decrease")
                class_weight_opt = st.selectbox("class_weight", ["None", "balanced"], index=0, key="class_dt_class_weight")
                model_params["class_weight"] = None if class_weight_opt == "None" else class_weight_opt
                model_params["ccp_alpha"] = st.number_input("ccp_alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="class_dt_ccp_alpha")

            elif model_name == "Random Forest Classifier":
                model_params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10, key="class_rf_n_estimators")
                model_params["criterion"] = st.selectbox("criterion", ["gini", "entropy", "log_loss"], index=0, key="class_rf_criterion")
                class_rf_max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=200, value=0, step=1, key="class_rf_max_depth")
                model_params["max_depth"] = None if class_rf_max_depth == 0 else int(class_rf_max_depth)
                model_params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=100, value=2, step=1, key="class_rf_min_samples_split")
                model_params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="class_rf_min_samples_leaf")
                model_params["min_weight_fraction_leaf"] = st.number_input("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="class_rf_min_weight_fraction_leaf")
                class_rf_max_features = st.selectbox("max_features", ["sqrt", "log2", "None"], index=0, key="class_rf_max_features")
                model_params["max_features"] = None if class_rf_max_features == "None" else class_rf_max_features
                class_rf_max_leaf_nodes = st.number_input("max_leaf_nodes (0 = None)", min_value=0, max_value=2000, value=0, step=1, key="class_rf_max_leaf_nodes")
                model_params["max_leaf_nodes"] = None if class_rf_max_leaf_nodes == 0 else int(class_rf_max_leaf_nodes)
                model_params["min_impurity_decrease"] = st.number_input("min_impurity_decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="class_rf_min_impurity_decrease")
                model_params["bootstrap"] = st.checkbox("bootstrap", value=True, key="class_rf_bootstrap")
                model_params["oob_score"] = st.checkbox("oob_score", value=False, key="class_rf_oob_score")
                model_params["n_jobs"] = st.number_input("n_jobs", min_value=-1, max_value=64, value=-1, step=1, key="class_rf_n_jobs")
                class_weight_opt = st.selectbox("class_weight", ["None", "balanced", "balanced_subsample"], index=0, key="class_rf_class_weight")
                model_params["class_weight"] = None if class_weight_opt == "None" else class_weight_opt
                model_params["ccp_alpha"] = st.number_input("ccp_alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="class_rf_ccp_alpha")
                class_rf_max_samples = st.number_input("max_samples (0 = None)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="class_rf_max_samples")
                model_params["max_samples"] = None if class_rf_max_samples == 0 else class_rf_max_samples
            
            test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="class_test")
            
            if st.button("Train Classification Model"):
                with st.spinner(f"Training {model_name}..."):
                    results = train_classification_model(
                        processed_df,
                        model_name,
                        target_col,
                        feature_cols,
                        test_size,
                        model_params=model_params,
                    )
                
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
                    plot_classification_visualizations(results, model_name)
    
    elif model_type == "Regression":
        st.subheader("Regression Models")
        st.write("Train a regression model to predict continuous targets.")
        
        numeric_cols = detect_numeric_columns(processed_df)
        
        if len(processed_df) < 2:
            st.warning("Regression requires at least 2 rows.")
        elif len(numeric_cols) < 2:
            st.warning("At least 2 numeric columns are required for regression (features + target).")
        else:
            target_col = st.selectbox(
                "Select target column (must be numeric)",
                options=numeric_cols,
                key="reg_target",
            )

            predictor_options = [col for col in numeric_cols if col != target_col]
            auto_reg_features = suggest_predictor_columns(
                processed_df,
                target_col,
                predictor_options,
                problem_type="regression",
            )
            if st.session_state.get("reg_target_prev") != target_col:
                st.session_state["reg_features"] = auto_reg_features
            st.session_state["reg_target_prev"] = target_col
            feature_cols = st.multiselect(
                "Select predictor columns (auto-selected)",
                options=predictor_options,
                default=auto_reg_features,
                key="reg_features",
                help="Choose the numeric input columns used to predict the target column.",
            )
            
            model_name = st.selectbox(
                "Select Regression Model",
                options=["Linear Regression", "KNN", "SVM", "Decision Tree", "Random Forest Regressor"],
                key="reg_model",
            )

            model_params = {}
            st.write("Model Hyperparameters")

            if model_name == "Linear Regression":
                model_params["fit_intercept"] = st.checkbox("fit_intercept", value=True, key="reg_lr_fit_intercept")
                model_params["copy_X"] = st.checkbox("copy_X", value=True, key="reg_lr_copy_x")
                reg_lr_n_jobs = st.number_input("n_jobs (0 = None)", min_value=0, max_value=64, value=0, step=1, key="reg_lr_n_jobs")
                model_params["n_jobs"] = None if reg_lr_n_jobs == 0 else int(reg_lr_n_jobs)
                model_params["positive"] = st.checkbox("positive", value=False, key="reg_lr_positive")

            elif model_name == "KNN":
                model_params["n_neighbors"] = st.number_input("n_neighbors", min_value=1, max_value=100, value=5, step=1, key="reg_knn_neighbors")
                model_params["weights"] = st.selectbox("weights", ["uniform", "distance"], index=0, key="reg_knn_weights")
                model_params["algorithm"] = st.selectbox("algorithm", ["auto", "ball_tree", "kd_tree", "brute"], index=0, key="reg_knn_algorithm")
                model_params["leaf_size"] = st.number_input("leaf_size", min_value=1, max_value=200, value=30, step=1, key="reg_knn_leaf_size")
                model_params["p"] = st.number_input("p", min_value=1, max_value=10, value=2, step=1, key="reg_knn_p")
                model_params["metric"] = st.selectbox("metric", ["minkowski", "euclidean", "manhattan", "chebyshev"], index=0, key="reg_knn_metric")

            elif model_name == "SVM":
                model_params["C"] = st.number_input("C", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1, key="reg_svm_c")
                model_params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid"], index=1, key="reg_svm_kernel")
                if model_params["kernel"] == "poly":
                    model_params["degree"] = st.number_input("degree", min_value=1, max_value=10, value=3, step=1, key="reg_svm_degree")
                else:
                    model_params["degree"] = 3
                model_params["gamma"] = st.selectbox("gamma", ["scale", "auto"], index=0, key="reg_svm_gamma")
                model_params["coef0"] = st.number_input("coef0", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="reg_svm_coef0")
                model_params["tol"] = st.number_input("tol", min_value=0.000001, max_value=1.0, value=0.001, step=0.0001, format="%.6f", key="reg_svm_tol")
                model_params["epsilon"] = st.number_input("epsilon", min_value=0.0001, max_value=10.0, value=0.1, step=0.01, key="reg_svm_epsilon")
                model_params["shrinking"] = st.checkbox("shrinking", value=True, key="reg_svm_shrinking")
                model_params["cache_size"] = st.number_input("cache_size", min_value=50.0, max_value=5000.0, value=200.0, step=50.0, key="reg_svm_cache")
                model_params["max_iter"] = st.number_input("max_iter (-1 for no limit)", min_value=-1, max_value=20000, value=-1, step=1, key="reg_svm_max_iter")

            elif model_name == "Decision Tree":
                model_params["criterion"] = st.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0, key="reg_dt_criterion")
                model_params["splitter"] = st.selectbox("splitter", ["best", "random"], index=0, key="reg_dt_splitter")
                reg_dt_max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=200, value=0, step=1, key="reg_dt_max_depth")
                model_params["max_depth"] = None if reg_dt_max_depth == 0 else int(reg_dt_max_depth)
                model_params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=100, value=2, step=1, key="reg_dt_min_samples_split")
                model_params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="reg_dt_min_samples_leaf")
                model_params["min_weight_fraction_leaf"] = st.number_input("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="reg_dt_min_weight_fraction_leaf")
                reg_dt_max_features = st.selectbox("max_features", ["None", "sqrt", "log2"], index=0, key="reg_dt_max_features")
                model_params["max_features"] = None if reg_dt_max_features == "None" else reg_dt_max_features
                reg_dt_max_leaf_nodes = st.number_input("max_leaf_nodes (0 = None)", min_value=0, max_value=2000, value=0, step=1, key="reg_dt_max_leaf_nodes")
                model_params["max_leaf_nodes"] = None if reg_dt_max_leaf_nodes == 0 else int(reg_dt_max_leaf_nodes)
                model_params["min_impurity_decrease"] = st.number_input("min_impurity_decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="reg_dt_min_impurity_decrease")
                model_params["ccp_alpha"] = st.number_input("ccp_alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="reg_dt_ccp_alpha")

            elif model_name == "Random Forest Regressor":
                model_params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=2000, value=100, step=10, key="reg_rf_n_estimators")
                model_params["criterion"] = st.selectbox("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"], index=0, key="reg_rf_criterion")
                reg_rf_max_depth = st.number_input("max_depth (0 = None)", min_value=0, max_value=200, value=0, step=1, key="reg_rf_max_depth")
                model_params["max_depth"] = None if reg_rf_max_depth == 0 else int(reg_rf_max_depth)
                model_params["min_samples_split"] = st.number_input("min_samples_split", min_value=2, max_value=100, value=2, step=1, key="reg_rf_min_samples_split")
                model_params["min_samples_leaf"] = st.number_input("min_samples_leaf", min_value=1, max_value=100, value=1, step=1, key="reg_rf_min_samples_leaf")
                model_params["min_weight_fraction_leaf"] = st.number_input("min_weight_fraction_leaf", min_value=0.0, max_value=0.5, value=0.0, step=0.01, key="reg_rf_min_weight_fraction_leaf")
                model_params["max_features"] = st.number_input("max_features (float)", min_value=0.1, max_value=1.0, value=1.0, step=0.1, key="reg_rf_max_features")
                reg_rf_max_leaf_nodes = st.number_input("max_leaf_nodes (0 = None)", min_value=0, max_value=2000, value=0, step=1, key="reg_rf_max_leaf_nodes")
                model_params["max_leaf_nodes"] = None if reg_rf_max_leaf_nodes == 0 else int(reg_rf_max_leaf_nodes)
                model_params["min_impurity_decrease"] = st.number_input("min_impurity_decrease", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="reg_rf_min_impurity_decrease")
                model_params["bootstrap"] = st.checkbox("bootstrap", value=True, key="reg_rf_bootstrap")
                model_params["oob_score"] = st.checkbox("oob_score", value=False, key="reg_rf_oob_score")
                model_params["n_jobs"] = st.number_input("n_jobs", min_value=-1, max_value=64, value=-1, step=1, key="reg_rf_n_jobs")
                model_params["ccp_alpha"] = st.number_input("ccp_alpha", min_value=0.0, max_value=1.0, value=0.0, step=0.001, format="%.3f", key="reg_rf_ccp_alpha")
                reg_rf_max_samples = st.number_input("max_samples (0 = None)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="reg_rf_max_samples")
                model_params["max_samples"] = None if reg_rf_max_samples == 0 else reg_rf_max_samples
            
            test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="reg_test")
            
            if st.button("Train Regression Model"):
                with st.spinner(f"Training {model_name}..."):
                    results = train_regression_model(
                        processed_df,
                        model_name,
                        target_col,
                        feature_cols,
                        test_size,
                        model_params=model_params,
                    )
                
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
                    plot_regression_visualizations(results, model_name)
    
    elif model_type == "Clustering":
        st.subheader("Clustering Models")
        st.write("Train a KMeans clustering model to group similar data points.")
        
        numeric_cols = detect_numeric_columns(processed_df)
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for clustering.")
        elif len(processed_df) < 2:
            st.warning("Clustering requires at least 2 rows.")
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
                value=min(3, len(processed_df)),
                key="n_clusters",
            )

            cluster_params = {}
            st.write("KMeans Hyperparameters")
            cluster_params["init"] = st.selectbox("init", ["k-means++", "random"], index=0, key="kmeans_init")
            cluster_params["n_init"] = st.number_input("n_init", min_value=1, max_value=100, value=10, step=1, key="kmeans_n_init")
            cluster_params["max_iter"] = st.number_input("max_iter", min_value=10, max_value=5000, value=300, step=10, key="kmeans_max_iter")
            cluster_params["tol"] = st.number_input("tol", min_value=0.000001, max_value=1.0, value=0.0001, step=0.0001, format="%.6f", key="kmeans_tol")
            cluster_params["algorithm"] = st.selectbox("algorithm", ["lloyd", "elkan"], index=0, key="kmeans_algorithm")
            
            if st.button("Train Clustering Model"):
                if not selected_cluster_cols:
                    st.warning("Select at least one numeric column for clustering.")
                else:
                    with st.spinner("Training KMeans..."):
                        results = train_clustering_model(
                            processed_df,
                            n_clusters,
                            selected_cluster_cols,
                            model_params=cluster_params,
                        )
                    
                    if "error" in results:
                        st.error(f"Error: {results['error']}")
                    else:
                        st.success("KMeans clustering completed successfully!")
                        col1, col2 = st.columns(2)
                        col1.metric("Number of Clusters", results['n_clusters'])
                        col2.metric("Inertia (WCSS)", f"{results['inertia']:.4f}")
                        
                        st.write(f"**Number of samples clustered:** {results['n_samples']}")
                        plot_clustering_visualizations(results, selected_cluster_cols)

    st.markdown("---")
    st.markdown("<div id='pipeline-summary'></div>", unsafe_allow_html=True)
    render_section_header("Pipeline Summary")
    if pipeline_steps:
        for i, step in enumerate(pipeline_steps, start=1):
            st.write(f"{i}. {step}")
    else:
        st.write("No preprocessing steps were applied yet.")

    st.markdown("---")
    st.markdown("<div id='notes'></div>", unsafe_allow_html=True)
    render_section_header("Notes")
    st.info(
        "This app uses pandas, numpy, seaborn, matplotlib and scikit-learn to provide a full data cleaning and exploration workflow." 
        "If your dataset has datetime-like text columns, the app will detect them and offer date feature extraction."
    )


if __name__ == "__main__":
    main()
