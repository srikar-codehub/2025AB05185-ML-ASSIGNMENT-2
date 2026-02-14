"""
Tab 1: Model Evaluation - Default evaluation using the sample Heart Disease dataset.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from utils import (
    get_model, evaluate_model, preprocess_data, load_training_data,
    MODEL_NAMES, SCALE_MODELS
)


def render(tab):
    """Render the Model Evaluation tab."""
    with tab:
        st.markdown("### Model Evaluation Results")
        st.write("Default evaluation using the sample Heart Disease dataset (918 samples).")

        df = load_training_data()
        target_col = 'HeartDisease'
        test_size = 0.20
        random_state = 42
        model_option = "Compare All Models"

        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.metric("Total Samples", df.shape[0])
        with info_col2:
            st.metric("Features", df.shape[1])
        with info_col3:
            st.metric("Test Size", "20%")

        st.markdown("---")
        st.markdown("### Training Results")

        df_processed = preprocess_data(df)

        if target_col in df_processed.columns:
            X = df_processed.drop(target_col, axis=1)
            y = df_processed[target_col]
        else:
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
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.write(f"**Training Set:** {len(X_train)} samples | **Test Set:** {len(X_test)} samples")

        if model_option == "Compare All Models":
            all_results = []
            progress_bar = st.progress(0)

            for idx, name in enumerate(MODEL_NAMES):
                model = get_model(name)
                if name in SCALE_MODELS:
                    metrics, _, _, _ = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
                else:
                    metrics, _, _, _ = evaluate_model(model, X_train, X_test, y_train, y_test)
                metrics['Model'] = name
                all_results.append(metrics)
                progress_bar.progress((idx + 1) / len(MODEL_NAMES))

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
                bars = ax.bar(range(len(MODEL_NAMES)), values, color='#000000', edgecolor='#000000')
                ax.set_title(f'{metric}', fontfamily='serif', fontsize=12, fontweight='bold')
                ax.set_ylabel(metric, fontfamily='serif')
                ax.set_ylim(0, 1.1)
                ax.set_xticks(range(len(MODEL_NAMES)))
                ax.set_xticklabels([m.split('(')[0].strip()[:8] for m in MODEL_NAMES], rotation=45, ha='right', fontfamily='serif')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontfamily='serif')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # Ranking & Analysis
            st.markdown("---")
            st.markdown("### Model Ranking & Analysis")

            weights = {'Accuracy': 0.15, 'AUC': 0.20, 'Precision': 0.15, 'Recall': 0.20, 'F1': 0.15, 'MCC': 0.15}
            results_df['Overall_Score'] = sum(results_df[metric] * weight for metric, weight in weights.items())
            results_df['Overall_Score'] = round(results_df['Overall_Score'], 4)

            ranking_df = results_df.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            ranking_df.index.name = 'Rank'

            rank_col1, rank_col2 = st.columns(2)

            with rank_col1:
                st.markdown("#### Overall Ranking")
                st.caption("Weighted score (AUC & Recall weighted higher)")

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

            # Model characteristics
            st.markdown("---")
            st.markdown("### Model Characteristics Summary")

            characteristics = {
                'Model': MODEL_NAMES,
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
            st.markdown(f"#### Model: {model_option}")

            model = get_model(model_option)

            if model_option in SCALE_MODELS:
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
