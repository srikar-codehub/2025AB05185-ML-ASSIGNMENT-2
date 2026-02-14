"""
Heart Disease Classification - Streamlit Web Application
This app demonstrates 6 ML classification models with evaluation metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

import warnings
warnings.filterwarnings('ignore')

# Page Configuration - No sidebar
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Minimalist Monochrome Theme + Hide Sidebar
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');

    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Global styles */
    .stApp {
        font-family: 'Source Serif 4', Georgia, serif;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', Georgia, serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    h1 {
        font-size: 3rem !important;
        border-bottom: 4px solid #000 !important;
        padding-bottom: 1rem !important;
    }

    /* Navigation tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        border-bottom: 2px solid #000 !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        border-bottom: none !important;
        background-color: #fff !important;
        color: #000 !important;
        font-family: 'Source Serif 4', serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #000 !important;
        color: #fff !important;
    }

    /* Remove rounded corners from all elements */
    .stButton > button {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        background-color: #000 !important;
        color: #fff !important;
        font-family: 'Source Serif 4', serif !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        font-weight: 600 !important;
        transition: all 0.1s !important;
    }

    .stButton > button:hover {
        background-color: #fff !important;
        color: #000 !important;
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 0 !important;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
    }

    /* DataFrames */
    .stDataFrame {
        border: 2px solid #000 !important;
    }

    /* Radio buttons - horizontal */
    .stRadio > div {
        flex-direction: row !important;
        gap: 1rem !important;
    }

    .stRadio > div > label {
        border: 2px solid #000 !important;
        padding: 0.5rem 1rem !important;
        background: #fff !important;
    }

    /* Dividers */
    hr {
        border: none !important;
        border-top: 2px solid #000 !important;
        margin: 2rem 0 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        border-radius: 0 !important;
        border: 2px solid #000 !important;
        background-color: #fff !important;
        color: #000 !important;
    }

    .stDownloadButton > button:hover {
        background-color: #000 !important;
        color: #fff !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #000 !important;
        border-radius: 0 !important;
        padding: 2rem !important;
    }

    /* Success/Warning/Error messages */
    .stAlert {
        border-radius: 0 !important;
        border-left: 4px solid #000 !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        border-radius: 0 !important;
    }

    /* Checkbox */
    .stCheckbox {
        border: 1px solid #000 !important;
        padding: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ============== HELPER FUNCTIONS ==============

def get_model(model_name):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "XGBoost (Ensemble)": XGBClassifier(n_estimators=100, random_state=42, max_depth=5,
                                            learning_rate=0.1, eval_metric='logloss')
    }
    return models.get(model_name)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1': round(f1_score(y_test, y_pred), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }

    return metrics, y_pred, y_pred_proba, confusion_matrix(y_test, y_pred)

def validate_data(df):
    issues = []
    warnings_list = []

    if df.empty or len(df) < 1:
        issues.append("Dataset is empty! Must have at least 1 row.")
        return issues, warnings_list, df

    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if len(cols_with_missing) > 0:
        warnings_list.append(f"Missing values found in: {', '.join(cols_with_missing.index.tolist())}")

    if 'HeartDisease' not in df.columns:
        potential_targets = [col for col in df.columns if 'target' in col.lower() or 'disease' in col.lower() or 'class' in col.lower()]
        if potential_targets:
            warnings_list.append(f"'HeartDisease' column not found. Using '{potential_targets[0]}' as target.")
        else:
            warnings_list.append("'HeartDisease' column not found. Using last column as target.")

    return issues, warnings_list, df

def clean_data(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    return df_clean

def preprocess_data(df):
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded

@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv('data/heart.csv')
        return df
    except Exception:
        np.random.seed(42)
        n_samples = 918
        data = {
            'Age': np.random.randint(28, 77, n_samples),
            'Sex': np.random.choice(['M', 'F'], n_samples),
            'ChestPainType': np.random.choice(['ATA', 'NAP', 'ASY', 'TA'], n_samples),
            'RestingBP': np.random.randint(80, 200, n_samples),
            'Cholesterol': np.random.randint(100, 600, n_samples),
            'FastingBS': np.random.choice([0, 1], n_samples),
            'RestingECG': np.random.choice(['Normal', 'ST', 'LVH'], n_samples),
            'MaxHR': np.random.randint(60, 202, n_samples),
            'ExerciseAngina': np.random.choice(['Y', 'N'], n_samples),
            'Oldpeak': np.round(np.random.uniform(-2.6, 6.2, n_samples), 1),
            'ST_Slope': np.random.choice(['Up', 'Flat', 'Down'], n_samples),
            'HeartDisease': np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
        }
        return pd.DataFrame(data)

@st.cache_resource
def train_model_for_prediction(model_name):
    df = load_training_data()
    df_processed = preprocess_data(df)

    target_col = 'HeartDisease' if 'HeartDisease' in df_processed.columns else df_processed.columns[-1]
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = get_model(model_name)

    scale_models = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian)"]
    if model_name in scale_models:
        model.fit(X_scaled, y)
    else:
        model.fit(X, y)

    return model, scaler, X.columns.tolist(), scale_models

# ============== MAIN APP ==============

# Title
st.title("Heart Disease Classification")
st.markdown("##### ML Models Comparison | Binary Classification")
st.markdown("---")

# Top Navigation using Tabs
tab1, tab2, tab3, tab4 = st.tabs(["MODEL EVALUATION", "RETRAIN", "SINGLE PREDICTION", "BULK PREDICTION"])

# ============== TAB 1: MODEL EVALUATION (Default Results) ==============
with tab1:
    st.markdown("### Model Evaluation Results")
    st.write("Default evaluation using the sample Heart Disease dataset (918 samples).")

    # Use default sample data
    df = load_training_data()
    target_col = 'HeartDisease'
    test_size = 0.20
    random_state = 42
    model_option = "Compare All Models"

    # Dataset info
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("Total Samples", df.shape[0])
    with info_col2:
        st.metric("Features", df.shape[1] - 1)
    with info_col3:
        st.metric("Test Size", "20%")

    st.markdown("---")
    st.markdown("### Training Results")

    # Preprocess and train
    df_processed = preprocess_data(df)

    # Handle target column after preprocessing
    if target_col in df_processed.columns:
        X = df_processed.drop(target_col, axis=1)
        y = df_processed[target_col]
    else:
        # Target might have been encoded, find it
        possible_targets = [c for c in df_processed.columns if target_col in c]
        if possible_targets:
            X = df_processed.drop(possible_targets[0], axis=1)
            y = df_processed[possible_targets[0]]
        else:
            X = df_processed.iloc[:, :-1]
            y = df_processed.iloc[:, -1]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # If stratify fails, try without it
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_models = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian)"]

    st.write(f"**Training Set:** {len(X_train)} samples | **Test Set:** {len(X_test)} samples")

    if model_option == "Compare All Models":
        all_results = []
        model_names = [
            "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)",
            "Naive Bayes (Gaussian)", "Random Forest (Ensemble)", "XGBoost (Ensemble)"
        ]

        progress_bar = st.progress(0)

        for idx, name in enumerate(model_names):
            model = get_model(name)
            if name in scale_models:
                metrics, _, _, _ = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
            else:
                metrics, _, _, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
            metrics['Model'] = name
            all_results.append(metrics)
            progress_bar.progress((idx + 1) / len(model_names))

        results_df = pd.DataFrame(all_results)
        results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

        st.markdown("#### Metrics Comparison Table")
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
                     use_container_width=True)

        st.markdown("#### Visual Comparison")
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            values = results_df[metric].values
            bars = ax.bar(range(len(model_names)), values, color='#000000', edgecolor='#000000')
            ax.set_title(f'{metric}', fontfamily='serif', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric, fontfamily='serif')
            ax.set_ylim(0, 1.1)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([m.split('(')[0].strip()[:8] for m in model_names], rotation=45, ha='right', fontfamily='serif')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontfamily='serif')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ============== DETAILED ANALYSIS & RANKING ==============
        st.markdown("---")
        st.markdown("### Model Ranking & Analysis")

        # Calculate overall score (weighted average of all metrics)
        weights = {'Accuracy': 0.15, 'AUC': 0.20, 'Precision': 0.15, 'Recall': 0.20, 'F1': 0.15, 'MCC': 0.15}
        results_df['Overall_Score'] = sum(results_df[metric] * weight for metric, weight in weights.items())
        results_df['Overall_Score'] = round(results_df['Overall_Score'], 4)

        # Create ranking
        ranking_df = results_df.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
        ranking_df.index = ranking_df.index + 1  # Start rank from 1
        ranking_df.index.name = 'Rank'

        rank_col1, rank_col2 = st.columns(2)

        with rank_col1:
            st.markdown("#### Overall Ranking")
            st.caption("Weighted score (AUC & Recall weighted higher)")

            # Display ranking table with color coding
            def highlight_rank(row):
                if row.name == 1:
                    return ['background-color: #000000; color: #ffffff'] * len(row)
                elif row.name == 2:
                    return ['background-color: #333333; color: #ffffff'] * len(row)
                elif row.name == 3:
                    return ['background-color: #666666; color: #ffffff'] * len(row)
                else:
                    return [''] * len(row)

            ranking_display = ranking_df[['Model', 'Overall_Score', 'Accuracy', 'AUC', 'F1']].copy()
            st.dataframe(ranking_display.style.apply(highlight_rank, axis=1), use_container_width=True)

        with rank_col2:
            st.markdown("#### Best Model by Metric")
            st.caption("Top performer for each evaluation metric")
            best_by_metric = {}
            for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']:
                best_idx = results_df[metric].idxmax()
                best_model = results_df.loc[best_idx, 'Model']
                best_value = results_df.loc[best_idx, metric]
                best_by_metric[metric] = f"{best_model.split('(')[0].strip()} ({best_value:.4f})"

            best_df = pd.DataFrame(list(best_by_metric.items()), columns=['Metric', 'Best Model (Score)'])
            st.dataframe(best_df, use_container_width=True, hide_index=True)

        # Best model recommendation
        best_model_name = ranking_df.iloc[0]['Model']
        best_score = ranking_df.iloc[0]['Overall_Score']
        second_best = ranking_df.iloc[1]['Model']
        second_score = ranking_df.iloc[1]['Overall_Score']

        st.markdown("---")
        st.markdown("### Recommendation")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.success(f"**RECOMMENDED: {best_model_name}**")
            st.write(f"Overall Score: **{best_score:.4f}** (out of 1.0)")

            # Generate specific recommendation based on the best model
            model_insights = {
                "Logistic Regression": "Interpretable, fast to train, provides probability outputs. Best for model explainability.",
                "Decision Tree": "Creates interpretable rules, handles mixed feature types. Consider pruning to avoid overfitting.",
                "K-Nearest Neighbors (KNN)": "Simple instance-based learner. Works well with normalized features but slow for large datasets.",
                "Naive Bayes (Gaussian)": "Fast training and prediction. Good for real-time applications with limited resources.",
                "Random Forest (Ensemble)": "Robust, handles overfitting well. Provides feature importance rankings.",
                "XGBoost (Ensemble)": "Powerful gradient boosting. Handles imbalanced data well with regularization."
            }

            st.write(model_insights.get(best_model_name, "Best overall performance on evaluation metrics."))

        with rec_col2:
            st.markdown("**Score Gap Analysis**")
            gap = best_score - second_score
            if gap > 0.05:
                st.write(f"Clear winner ({gap:.4f} ahead)")
            elif gap > 0.02:
                st.write(f"Moderate advantage ({gap:.4f})")
            else:
                st.write(f"Close competition ({gap:.4f})")
                st.write(f"Consider {second_best} as alternative")

            st.markdown("")
            st.markdown("**Runner-up:**")
            st.write(f"{second_best}")
            st.write(f"Score: {second_score:.4f}")

        # Use case recommendations
        st.markdown("---")
        st.markdown("### Use Case Recommendations")

        use_col1, use_col2, use_col3 = st.columns(3)

        # Find best model for each use case
        high_recall_model = results_df.loc[results_df['Recall'].idxmax(), 'Model']
        high_precision_model = results_df.loc[results_df['Precision'].idxmax(), 'Model']
        balanced_model = results_df.loc[results_df['F1'].idxmax(), 'Model']

        with use_col1:
            st.markdown("**High Sensitivity (Recall)**")
            st.markdown("*Minimize missed diagnoses*")
            st.write(f"Use: **{high_recall_model.split('(')[0].strip()}**")
            st.write(f"Recall: {results_df.loc[results_df['Recall'].idxmax(), 'Recall']:.4f}")
            st.caption("Best when false negatives are costly (missing a disease case)")

        with use_col2:
            st.markdown("**High Precision**")
            st.markdown("*Minimize false alarms*")
            st.write(f"Use: **{high_precision_model.split('(')[0].strip()}**")
            st.write(f"Precision: {results_df.loc[results_df['Precision'].idxmax(), 'Precision']:.4f}")
            st.caption("Best when false positives are costly (unnecessary treatments)")

        with use_col3:
            st.markdown("**Balanced (F1 Score)**")
            st.markdown("*Best trade-off*")
            st.write(f"Use: **{balanced_model.split('(')[0].strip()}**")
            st.write(f"F1: {results_df.loc[results_df['F1'].idxmax(), 'F1']:.4f}")
            st.caption("Best for general-purpose classification")

        # Model characteristics table
        st.markdown("---")
        st.markdown("### Model Characteristics Summary")

        characteristics = {
            'Model': model_names,
            'Type': ['Linear', 'Tree-based', 'Instance-based', 'Probabilistic', 'Ensemble', 'Ensemble'],
            'Interpretability': ['High', 'High', 'Low', 'Medium', 'Medium', 'Low'],
            'Training Speed': ['Fast', 'Fast', 'Fast', 'Fast', 'Medium', 'Medium'],
            'Prediction Speed': ['Fast', 'Fast', 'Slow', 'Fast', 'Medium', 'Fast'],
            'Handles Outliers': ['No', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
            'Feature Scaling': ['Required', 'Not Required', 'Required', 'Not Required', 'Not Required', 'Not Required']
        }
        char_df = pd.DataFrame(characteristics)
        st.dataframe(char_df, use_container_width=True, hide_index=True)

    else:
        # Single model evaluation
        st.markdown(f"#### Model: {model_option}")

        model = get_model(model_option)

        if model_option in scale_models:
            metrics, y_pred, y_pred_proba, cm = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            metrics, y_pred, y_pred_proba, cm = evaluate_model(model, X_train, X_test, y_train, y_test)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        with col2:
            st.metric("AUC", f"{metrics['AUC']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col5:
            st.metric("F1 Score", f"{metrics['F1']:.4f}")
        with col6:
            st.metric("MCC", f"{metrics['MCC']:.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
                        xticklabels=['No Disease', 'Disease'],
                        yticklabels=['No Disease', 'Disease'], ax=ax,
                        linewidths=2, linecolor='black')
            ax.set_title('Confusion Matrix', fontfamily='serif', fontweight='bold')
            ax.set_ylabel('Actual', fontfamily='serif')
            ax.set_xlabel('Predicted', fontfamily='serif')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

# ============== TAB 2: RETRAIN ==============
with tab2:
    st.markdown("### Retrain Models")
    st.write("Upload your own dataset or use the sample dataset to train models with custom settings.")

    st.markdown("---")

    # Row 1: Sample Download + Upload Option
    retrain_col_a, retrain_col_b = st.columns(2)

    with retrain_col_a:
        st.markdown("#### Download Sample Dataset")
        st.caption("Get the Heart Disease dataset to test training")
        try:
            sample_df_retrain = load_training_data()
            sample_csv_retrain = sample_df_retrain.to_csv(index=False)
            st.download_button(
                label="DOWNLOAD HEART.CSV",
                data=sample_csv_retrain,
                file_name="heart_disease_sample.csv",
                mime="text/csv",
                key="download_sample_retrain"
            )
        except Exception:
            st.warning("Sample data unavailable")

    with retrain_col_b:
        st.markdown("#### Or Use Sample Data")
        st.caption("Train models using the built-in dataset")
        use_sample_retrain = st.checkbox("Use Sample Data", value=True, key="retrain_sample")

    st.markdown("---")

    # Row 2: File Upload
    st.markdown("#### Upload Your Dataset")
    uploaded_file_retrain = st.file_uploader(
        "Upload CSV file in the same format as sample (with HeartDisease column)",
        type="csv",
        key="retrain_upload"
    )

    # Load data for retrain
    df_retrain = None
    if use_sample_retrain:
        df_retrain = load_training_data()
        st.success("Using sample Heart Disease dataset (918 rows x 12 columns)")
    elif uploaded_file_retrain is not None:
        try:
            df_retrain = pd.read_csv(uploaded_file_retrain)
            issues, warnings_list, df_retrain = validate_data(df_retrain)

            if issues:
                for issue in issues:
                    st.error(issue)
                st.stop()

            if warnings_list:
                for warning in warnings_list:
                    st.warning(warning)

            if df_retrain.isnull().sum().sum() > 0:
                if st.checkbox("Auto-clean missing values", value=True, key="retrain_clean"):
                    df_retrain = clean_data(df_retrain)
                    st.info("Missing values filled automatically.")

            st.success(f"Dataset loaded: {df_retrain.shape[0]} rows x {df_retrain.shape[1]} columns")

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    else:
        st.info("Upload a CSV file or check 'Use Sample Data' above to begin.")
        st.stop()

    st.markdown("---")

    # Row 3: Retraining Configuration
    st.markdown("#### Training Configuration")

    retrain_cfg1, retrain_cfg2, retrain_cfg3 = st.columns(3)

    with retrain_cfg1:
        test_size_retrain = st.selectbox(
            "Test Size",
            options=["10%", "20%", "30%", "40%"],
            index=1,
            key="retrain_test_size"
        )
        test_size_retrain = int(test_size_retrain.replace("%", "")) / 100

    with retrain_cfg2:
        random_state_retrain = st.number_input("Random Seed", min_value=0, max_value=999, value=42, key="retrain_seed")

    with retrain_cfg3:
        model_option_retrain = st.selectbox(
            "Model",
            [
                "Compare All Models",
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ],
            key="retrain_model"
        )
        model_map_retrain = {
            "KNN": "K-Nearest Neighbors (KNN)",
            "Naive Bayes": "Naive Bayes (Gaussian)",
            "Random Forest": "Random Forest (Ensemble)",
            "XGBoost": "XGBoost (Ensemble)"
        }
        if model_option_retrain in model_map_retrain:
            model_option_retrain = model_map_retrain[model_option_retrain]

    st.markdown("")
    if st.button("TRAIN MODELS", type="primary", key="retrain_btn", use_container_width=True):
        target_col_retrain = 'HeartDisease'

        # Dataset preview - both in fixed height containers
        preview_r1, preview_r2 = st.columns(2)

        PREVIEW_HEIGHT = 350

        with preview_r1:
            st.markdown("#### Data Sample")
            st.caption(f"Rows: {df_retrain.shape[0]} | Columns: {df_retrain.shape[1]}")
            st.dataframe(df_retrain.head(10), use_container_width=True, height=PREVIEW_HEIGHT)

        with preview_r2:
            st.markdown("#### Target Distribution")
            vc = df_retrain[target_col_retrain].value_counts()
            st.caption(f"Class 0: {vc.get(0, 0)} | Class 1: {vc.get(1, 0)}")
            # Create chart to match table height (accounting for ~50px header/caption)
            chart_height = (PREVIEW_HEIGHT - 50) / 100  # Convert px to inches approx
            fig_r, ax_r = plt.subplots(figsize=(5, chart_height))
            colors_r = ['#000000', '#666666']
            vc.plot(kind='bar', color=colors_r[:len(vc)], ax=ax_r, edgecolor='black', linewidth=2)
            ax_r.set_xlabel('')
            ax_r.set_ylabel('Count', fontfamily='serif', fontsize=11)
            ax_r.spines['top'].set_visible(False)
            ax_r.spines['right'].set_visible(False)
            ax_r.tick_params(axis='both', labelsize=10)
            plt.xticks(rotation=0)
            for i, v in enumerate(vc.values):
                ax_r.text(i, v + 5, str(v), ha='center', fontfamily='serif', fontweight='bold', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_r, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("### Training Results")

        # Preprocess and train
        df_processed_retrain = preprocess_data(df_retrain)

        if target_col_retrain in df_processed_retrain.columns:
            X_r = df_processed_retrain.drop(target_col_retrain, axis=1)
            y_r = df_processed_retrain[target_col_retrain]
        else:
            X_r = df_processed_retrain.iloc[:, :-1]
            y_r = df_processed_retrain.iloc[:, -1]

        try:
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_r, y_r, test_size=test_size_retrain, random_state=random_state_retrain, stratify=y_r
            )
        except ValueError:
            X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                X_r, y_r, test_size=test_size_retrain, random_state=random_state_retrain
            )

        scaler_r = StandardScaler()
        X_train_scaled_r = scaler_r.fit_transform(X_train_r)
        X_test_scaled_r = scaler_r.transform(X_test_r)

        scale_models_r = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian)"]

        st.write(f"**Training Set:** {len(X_train_r)} samples | **Test Set:** {len(X_test_r)} samples")

        if model_option_retrain == "Compare All Models":
            all_results_r = []
            model_names_r = [
                "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)",
                "Naive Bayes (Gaussian)", "Random Forest (Ensemble)", "XGBoost (Ensemble)"
            ]

            progress_r = st.progress(0)
            for idx, name in enumerate(model_names_r):
                model_r = get_model(name)
                if name in scale_models_r:
                    metrics_r, _, _, _ = evaluate_model(model_r, X_train_scaled_r, X_test_scaled_r, y_train_r, y_test_r)
                else:
                    metrics_r, _, _, _ = evaluate_model(model_r, X_train_r, X_test_r, y_train_r, y_test_r)
                metrics_r['Model'] = name
                all_results_r.append(metrics_r)
                progress_r.progress((idx + 1) / len(model_names_r))

            results_df_r = pd.DataFrame(all_results_r)
            results_df_r = results_df_r[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

            st.markdown("#### Metrics Comparison Table")
            st.dataframe(results_df_r.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
                         use_container_width=True)

            # Find best model
            results_df_r['Overall'] = (results_df_r['Accuracy'] + results_df_r['AUC'] + results_df_r['F1']) / 3
            best_r = results_df_r.loc[results_df_r['Overall'].idxmax()]
            st.success(f"**Best Model: {best_r['Model']}** (Avg Score: {best_r['Overall']:.4f})")

        else:
            model_r = get_model(model_option_retrain)
            if model_option_retrain in scale_models_r:
                metrics_r, y_pred_r, _, cm_r = evaluate_model(model_r, X_train_scaled_r, X_test_scaled_r, y_train_r, y_test_r)
            else:
                metrics_r, y_pred_r, _, cm_r = evaluate_model(model_r, X_train_r, X_test_r, y_train_r, y_test_r)

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{metrics_r['Accuracy']:.4f}")
            m2.metric("AUC", f"{metrics_r['AUC']:.4f}")
            m3.metric("Precision", f"{metrics_r['Precision']:.4f}")
            m4.metric("Recall", f"{metrics_r['Recall']:.4f}")
            m5.metric("F1", f"{metrics_r['F1']:.4f}")
            m6.metric("MCC", f"{metrics_r['MCC']:.4f}")

# ============== TAB 3: SINGLE PREDICTION ==============
with tab3:
    st.markdown("### Single Patient Prediction")
    st.write("Enter patient details below to predict heart disease risk.")

    # Model selection at top
    model_option_single = st.selectbox(
        "Select Model for Prediction",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes (Gaussian)",
            "Random Forest (Ensemble)",
            "XGBoost (Ensemble)"
        ],
        key="single_model"
    )

    st.markdown("---")

    # Load trained model
    model, scaler, feature_cols, scale_models = train_model_for_prediction(model_option_single)

    # Create input form
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50, key="s_age")
        sex = st.selectbox("Sex", ["M", "F"], key="s_sex")
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"], key="s_cp")
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, key="s_bp")

    with col2:
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=700, value=200, key="s_chol")
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], key="s_fbs")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], key="s_ecg")
        max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150, key="s_hr")

    with col3:
        exercise_angina = st.selectbox("Exercise Induced Angina", ["N", "Y"], key="s_angina")
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-5.0, max_value=10.0, value=0.0, step=0.1, key="s_oldpeak")
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="s_slope")

    st.markdown("")

    # Predict button
    if st.button("PREDICT HEART DISEASE RISK", type="primary", key="predict_single"):
        input_data = pd.DataFrame({
            'Age': [age], 'Sex': [sex], 'ChestPainType': [chest_pain],
            'RestingBP': [resting_bp], 'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs], 'RestingECG': [resting_ecg],
            'MaxHR': [max_hr], 'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak], 'ST_Slope': [st_slope]
        })

        input_processed = preprocess_data(input_data)

        for col in feature_cols:
            if col not in input_processed.columns:
                input_processed[col] = 0
        input_processed = input_processed[feature_cols]

        if model_option_single in scale_models:
            input_scaled = scaler.transform(input_processed)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
        else:
            prediction = model.predict(input_processed)[0]
            probability = model.predict_proba(input_processed)[0]

        st.markdown("---")
        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("**PREDICTION: HEART DISEASE DETECTED**")
            else:
                st.success("**PREDICTION: NO HEART DISEASE**")

        with col2:
            st.write("**Confidence:**")
            st.write(f"- No Disease: {probability[0]*100:.2f}%")
            st.write(f"- Heart Disease: {probability[1]*100:.2f}%")

        fig, ax = plt.subplots(figsize=(8, 1.5))
        ax.barh(['Risk'], [probability[1]], color='#000000' if prediction == 1 else '#666666', height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability of Heart Disease', fontfamily='serif')
        ax.axvline(x=0.5, color='#999999', linestyle='--', linewidth=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ============== TAB 4: BULK PREDICTION ==============
with tab4:
    st.markdown("### Bulk Prediction Mode")
    st.write("Upload a CSV file with patient data to get predictions for all rows.")

    st.markdown("---")

    # Sample CSV download and Model selection in same row
    bulk_col1, bulk_col2 = st.columns(2)

    with bulk_col1:
        st.markdown("#### Download Test Data")
        st.caption("Sample CSV with 10 patients for testing")

        # Create sample data for bulk prediction (without target column)
        sample_bulk_data = pd.DataFrame({
            'Age': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57],
            'Sex': ['M', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
            'ChestPainType': ['ASY', 'NAP', 'ATA', 'ASY', 'ASY', 'NAP', 'ATA', 'ATA', 'NAP', 'ASY'],
            'RestingBP': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150],
            'Cholesterol': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168],
            'FastingBS': [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'RestingECG': ['LVH', 'Normal', 'LVH', 'Normal', 'Normal', 'Normal', 'LVH', 'Normal', 'Normal', 'Normal'],
            'MaxHR': [150, 187, 172, 178, 163, 148, 153, 173, 162, 174],
            'ExerciseAngina': ['N', 'N', 'N', 'N', 'Y', 'N', 'N', 'N', 'N', 'N'],
            'Oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 0.4, 1.3, 0.0, 0.5, 1.6],
            'ST_Slope': ['Down', 'Down', 'Up', 'Up', 'Up', 'Flat', 'Flat', 'Up', 'Up', 'Up']
        })

        sample_bulk_csv = sample_bulk_data.to_csv(index=False)
        st.download_button(
            label="DOWNLOAD TEST CSV (10 rows)",
            data=sample_bulk_csv,
            file_name="bulk_prediction_test.csv",
            mime="text/csv",
            key="download_bulk_sample"
        )

    with bulk_col2:
        st.markdown("#### Select Model")
        st.caption("Choose ML model for predictions")
        model_option_bulk = st.selectbox(
            "Model",
            [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors (KNN)",
                "Naive Bayes (Gaussian)",
                "Random Forest (Ensemble)",
                "XGBoost (Ensemble)"
            ],
            key="bulk_model",
            label_visibility="collapsed"
        )

    st.markdown("---")

    # File uploader
    st.markdown("#### Upload Your Data")
    uploaded_file_bulk = st.file_uploader(
        "Upload CSV file with patient data (without HeartDisease column)",
        type="csv",
        key="bulk_upload"
    )

    if uploaded_file_bulk is not None:
        try:
            df_predict = pd.read_csv(uploaded_file_bulk)

            if df_predict.empty:
                st.error("Dataset is empty! Must have at least 1 row.")
                st.stop()

            st.success(f"Uploaded {len(df_predict)} rows for prediction.")

            st.markdown("#### Data Preview")
            st.dataframe(df_predict.head(10), use_container_width=True)

            if df_predict.isnull().sum().sum() > 0:
                if st.checkbox("Auto-clean missing values", value=True, key="bulk_clean"):
                    df_predict = clean_data(df_predict)
                    st.info("Missing values filled with median/mode.")

            if 'HeartDisease' in df_predict.columns:
                df_predict = df_predict.drop('HeartDisease', axis=1)

            if st.button("GENERATE PREDICTIONS", type="primary", key="predict_bulk"):
                model, scaler, feature_cols, scale_models = train_model_for_prediction(model_option_bulk)

                df_processed = preprocess_data(df_predict)

                for col in feature_cols:
                    if col not in df_processed.columns:
                        df_processed[col] = 0
                df_processed = df_processed[feature_cols]

                if model_option_bulk in scale_models:
                    df_scaled = scaler.transform(df_processed)
                    predictions = model.predict(df_scaled)
                    probabilities = model.predict_proba(df_scaled)[:, 1]
                else:
                    predictions = model.predict(df_processed)
                    probabilities = model.predict_proba(df_processed)[:, 1]

                results_df = df_predict.copy()
                results_df['Predicted_HeartDisease'] = predictions
                results_df['Prediction_Probability'] = np.round(probabilities, 4)
                results_df['Risk_Level'] = results_df['Prediction_Probability'].apply(
                    lambda x: 'High Risk' if x >= 0.5 else 'Low Risk'
                )

                st.markdown("---")
                st.markdown("### Prediction Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(results_df))
                with col2:
                    st.metric("Predicted with Disease", int(predictions.sum()))
                with col3:
                    st.metric("Predicted Healthy", int(len(predictions) - predictions.sum()))

                st.dataframe(results_df, use_container_width=True)

                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="DOWNLOAD PREDICTIONS AS CSV",
                    data=csv,
                    file_name=f"predictions_{model_option_bulk.replace(' ', '_').lower()}.csv",
                    mime="text/csv"
                )

                st.markdown("#### Prediction Distribution")
                col1, col2 = st.columns(2)

                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    counts = results_df['Predicted_HeartDisease'].value_counts()
                    ax.pie(counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                           colors=['#666666', '#000000'], wedgeprops={'linewidth': 2, 'edgecolor': 'white'})
                    ax.set_title('Prediction Distribution', fontfamily='serif', fontweight='bold')
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    risk_counts = results_df['Risk_Level'].value_counts()
                    colors_risk = {'Low Risk': '#666666', 'High Risk': '#000000'}
                    bars = ax.bar(risk_counts.index, risk_counts.values,
                                  color=[colors_risk.get(x, '#000') for x in risk_counts.index],
                                  edgecolor='black', linewidth=2)
                    ax.set_title('Risk Level Distribution', fontfamily='serif', fontweight='bold')
                    ax.set_xlabel('Risk Level', fontfamily='serif')
                    ax.set_ylabel('Count', fontfamily='serif')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)
                    plt.close()

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to get bulk predictions.")

        st.markdown("#### Expected CSV Format")
        sample_df = pd.DataFrame({
            'Age': [50, 60],
            'Sex': ['M', 'F'],
            'ChestPainType': ['ATA', 'ASY'],
            'RestingBP': [120, 140],
            'Cholesterol': [200, 250],
            'FastingBS': [0, 1],
            'RestingECG': ['Normal', 'ST'],
            'MaxHR': [150, 130],
            'ExerciseAngina': ['N', 'Y'],
            'Oldpeak': [0.0, 1.5],
            'ST_Slope': ['Up', 'Flat']
        })
        st.dataframe(sample_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**About this App** | Heart Disease Classification using 6 ML Models

**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost

**Dataset:** UCI Heart Disease / Kaggle Heart Failure Prediction
""")
st.markdown("*Developed for ML Assignment - BITS Pilani*")
