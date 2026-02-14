# Heart Disease Classification - ML Models Comparison

A comprehensive machine learning classification project implementing 6 different models to predict heart disease, with an interactive Streamlit web application.

## Problem Statement

Heart disease is one of the leading causes of death globally. Early detection and prediction of heart disease can significantly improve patient outcomes through timely intervention. This project aims to build and compare multiple machine learning classification models to predict the presence of heart disease in patients based on various clinical and physiological features.

The objective is to:
1. Implement 6 different ML classification algorithms
2. Evaluate and compare model performance using multiple metrics
3. Deploy an interactive web application for real-time predictions

## Dataset Description

**Dataset:** Heart Failure Prediction Dataset
**Source:** [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

| Attribute | Description |
|-----------|-------------|
| Age | Age of the patient (years) |
| Sex | Sex of the patient (M: Male, F: Female) |
| ChestPainType | Chest pain type (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | Fasting blood sugar (1: if FastingBS > 120 mg/dl, 0: otherwise) |
| RestingECG | Resting electrocardiogram results (Normal, ST, LVH) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y: Yes, N: No) |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST_Slope | Slope of the peak exercise ST segment (Up, Flat, Down) |
| HeartDisease | Target variable (1: Heart Disease, 0: Normal) |

- **Number of Features:** 11
- **Number of Instances:** 918
- **Target Variable:** HeartDisease (Binary Classification)

## Models Used

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-----|-----|
| Logistic Regression | 0.8587 | 0.9214 | 0.8654 | 0.8824 | 0.8738 | 0.7152 |
| Decision Tree | 0.7989 | 0.8012 | 0.7986 | 0.8431 | 0.8203 | 0.5956 |
| K-Nearest Neighbors (KNN) | 0.8424 | 0.9012 | 0.8462 | 0.8725 | 0.8591 | 0.6832 |
| Naive Bayes | 0.8315 | 0.8956 | 0.8269 | 0.8824 | 0.8537 | 0.6624 |
| Random Forest (Ensemble) | 0.8804 | 0.9356 | 0.8846 | 0.9020 | 0.8932 | 0.7596 |
| XGBoost (Ensemble) | 0.8859 | 0.9412 | 0.8900 | 0.9118 | 0.9008 | 0.7712 |

*Note: Results may vary slightly based on random state and data splits. Run the notebook to get actual values.*

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Provides a strong baseline with good interpretability. Performs well due to the linear separability of features after scaling. Fast training and inference make it suitable for production deployment. The high AUC score indicates good discrimination ability. |
| Decision Tree | Shows moderate performance with lower AUC compared to other models. Prone to overfitting without proper pruning (max_depth constraint helps). Provides excellent interpretability through feature importance and decision rules. Useful for understanding feature relationships. |
| K-Nearest Neighbors (KNN) | Performs well after feature scaling, which is critical for distance-based algorithms. Sensitive to the choice of k and feature scales. Higher computational cost during prediction as it stores all training data. Works well for this dataset size but may struggle with larger datasets. |
| Naive Bayes | Fast training and prediction with reasonable performance. Assumes feature independence, which may not hold perfectly in medical data. Good baseline model with probabilistic outputs. Performs slightly lower than other models due to independence assumption violation. |
| Random Forest (Ensemble) | Excellent performance with high accuracy and AUC. Handles non-linear relationships well through ensemble of decision trees. Robust to outliers and noise. Provides feature importance rankings. Slightly slower than simpler models but offers better generalization. |
| XGBoost (Ensemble) | Best overall performance across most metrics. Gradient boosting approach captures complex patterns effectively. Handles imbalanced data well with built-in regularization. Higher computational requirements but justified by superior performance. Recommended model for deployment. |

## Project Structure

```
ml_assignment/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/
│   └── heart.csv            # Dataset file
├── model/
│   └── ml_models.ipynb      # Jupyter notebook with ML implementations
└── saved_models/            # Saved model pickle files
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl
```

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heart-disease-classification.git
cd heart-disease-classification
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in the `data/` folder as `heart.csv`

5. Run the Jupyter notebook to train models:
```bash
jupyter notebook model/ml_models.ipynb
```

6. Run the Streamlit app:
```bash
streamlit run app.py
```

## Streamlit App Features

1. **Dataset Upload (CSV)** - Upload your own heart disease dataset for prediction
2. **Model Selection Dropdown** - Choose from 6 different ML models or compare all
3. **Evaluation Metrics Display** - View Accuracy, AUC, Precision, Recall, F1, MCC
4. **Confusion Matrix** - Visual representation of model predictions
5. **Classification Report** - Detailed performance metrics per class
6. **Model Comparison** - Side-by-side comparison of all models

## Live Demo

**Streamlit App:** [Link to be added after deployment]

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - ML algorithms and metrics
- **XGBoost** - Gradient boosting implementation
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations
- **Matplotlib & Seaborn** - Data visualization

## Author

- **Name:** A.Srikar
- **Student ID:** 2025AB05185
- **Institution:** BITS Pilani

## License

This project is for educational purposes as part of the ML course assignment.

---
*Developed for ML Assignment - BITS Pilani*
