"""
Tab 3: Single Prediction - Enter patient details to predict heart disease risk.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import preprocess_data, train_model_for_prediction, MODEL_NAMES


def render(tab):
    """Render the Single Prediction tab."""
    with tab:
        st.markdown("### Single Patient Prediction")
        st.write("Enter patient details below to predict heart disease risk.")

        model_option_single = st.selectbox(
            "Select Model for Prediction",
            MODEL_NAMES,
            key="single_model"
        )

        st.markdown("---")

        model, scaler, feature_cols, scale_models = train_model_for_prediction(model_option_single)

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
