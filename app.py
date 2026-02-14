"""
Heart Disease Classification - Streamlit Web Application
This app demonstrates 6 ML classification models with evaluation metrics.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from styles import CUSTOM_CSS
import tab_evaluation
import tab_retrain
import tab_prediction
import tab_bulk

# Page Configuration
st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Loading notice for free tier
st.info("Please wait while the app loads. Switching between tabs may take a moment as this is hosted on Streamlit's free tier.")

# Title
st.title("Heart Disease Classification")
st.markdown("##### ML Models Comparison | Binary Classification")
st.markdown("---")

# Top Navigation using Tabs
tab1, tab2, tab3, tab4 = st.tabs(["MODEL EVALUATION", "RETRAIN", "SINGLE PREDICTION", "BULK PREDICTION"])

# Render each tab
tab_evaluation.render(tab1)
tab_retrain.render(tab2)
tab_prediction.render(tab3)
tab_bulk.render(tab4)

# Footer
st.markdown("---")
st.markdown("""
**About this App** | Heart Disease Classification using 6 ML Models

**Models:** Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost

**Dataset:** UCI Heart Disease / Kaggle Heart Failure Prediction
""")
st.markdown("*Developed for ML Assignment - BITS Pilani*")
