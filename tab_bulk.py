"""
Tab 4: Bulk Prediction - Upload CSV to get predictions for all rows.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    clean_data, preprocess_data, train_model_for_prediction, MODEL_NAMES
)


def render(tab):
    """Render the Bulk Prediction tab."""
    with tab:
        st.markdown("### Bulk Prediction Mode")
        st.write("Upload a CSV file with patient data to get predictions for all rows.")

        st.markdown("---")

        bulk_col1, bulk_col2 = st.columns(2)

        with bulk_col1:
            st.markdown("#### Download Test Data")
            st.caption("Sample CSV with 10 patients for testing")

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
                MODEL_NAMES,
                key="bulk_model",
                label_visibility="collapsed"
            )

        st.markdown("---")

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
