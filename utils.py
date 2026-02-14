"""
Utility functions for Heart Disease Classification App.
Includes model creation, evaluation, data validation, and preprocessing.
"""

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix
)


def get_model(model_name):
    """Return an ML model instance by name."""
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
    """Train a model and return evaluation metrics."""
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
    """Validate uploaded dataset and return issues/warnings."""
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
    """Fill missing values with median (numeric) or mode (categorical)."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    return df_clean


def preprocess_data(df):
    """One-hot encode categorical columns."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_encoded


@st.cache_data
def load_training_data():
    """Load the heart disease dataset from CSV or generate sample data."""
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
    """Train a model on the full dataset for prediction use."""
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


# List of all model names
MODEL_NAMES = [
    "Logistic Regression", "Decision Tree", "K-Nearest Neighbors (KNN)",
    "Naive Bayes (Gaussian)", "Random Forest (Ensemble)", "XGBoost (Ensemble)"
]

# Models that require feature scaling
SCALE_MODELS = ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Naive Bayes (Gaussian)"]
