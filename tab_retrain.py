"""
Tab 2: Retrain - Upload datasets and retrain models with custom settings.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    get_model, evaluate_model, validate_data, clean_data,
    preprocess_data, load_training_data,
    MODEL_NAMES, SCALE_MODELS
)


def render(tab):
    """Render the Retrain tab."""
    with tab:
        st.markdown("### Retrain Models")
        st.write("Upload your own dataset or use the sample dataset to train models with custom settings.")

        st.markdown("---")

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

        st.markdown("#### Upload Your Dataset")
        uploaded_file_retrain = st.file_uploader(
            "Upload CSV file in the same format as sample (with HeartDisease column)",
            type="csv",
            key="retrain_upload"
        )

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
                chart_height = (PREVIEW_HEIGHT - 50) / 100
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

            st.write(f"**Training Set:** {len(X_train_r)} samples | **Test Set:** {len(X_test_r)} samples")

            if model_option_retrain == "Compare All Models":
                all_results_r = []
                progress_r = st.progress(0)
                for idx, name in enumerate(MODEL_NAMES):
                    model_r = get_model(name)
                    if name in SCALE_MODELS:
                        metrics_r, _, _, _ = evaluate_model(model_r, X_train_scaled_r, X_test_scaled_r, y_train_r, y_test_r)
                    else:
                        metrics_r, _, _, _ = evaluate_model(model_r, X_train_r, X_test_r, y_train_r, y_test_r)
                    metrics_r['Model'] = name
                    all_results_r.append(metrics_r)
                    progress_r.progress((idx + 1) / len(MODEL_NAMES))

                results_df_r = pd.DataFrame(all_results_r)
                results_df_r = results_df_r[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]

                st.markdown("#### Metrics Comparison Table")
                st.dataframe(results_df_r.style.highlight_max(axis=0, subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']),
                             use_container_width=True)

                results_df_r['Overall'] = (results_df_r['Accuracy'] + results_df_r['AUC'] + results_df_r['F1']) / 3
                best_r = results_df_r.loc[results_df_r['Overall'].idxmax()]
                st.success(f"**Best Model: {best_r['Model']}** (Avg Score: {best_r['Overall']:.4f})")

            else:
                model_r = get_model(model_option_retrain)
                if model_option_retrain in SCALE_MODELS:
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
