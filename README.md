# Breast Cancer Prediction with Logistic Regression 🎗️

This project demonstrates a machine learning workflow to predict breast cancer malignancy using a **Logistic Regression** model. The process involves loading a dataset, preprocessing the data, training a model, and evaluating its performance.

## 💾 Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is available through scikit-learn. It contains **30 features** computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The goal is to predict the **diagnosis**, which is either **malignant** (cancerous) or **benign** (non-cancerous).

- **Features**: `mean radius`, `mean texture`, `mean perimeter`, `mean area`, etc.
- **Target**: `malignant` (0) or `benign` (1)

## 🛠️ Process

1.  **Data Loading and Exploration**:
    -   The `load_breast_cancer` function from scikit-learn is used to load the dataset.
    -   A pandas DataFrame is created to organize the features and the target variable.
    -   The shape of the DataFrame is `(569, 31)`, indicating 569 samples and 31 columns (30 features + 1 target).

2.  **Data Splitting**:
    -   The data is split into **training** and **testing** sets using `train_test_split`.
    -   **80%** of the data is allocated for training, and **20%** is used for testing.
    -   The `random_state` is set for reproducibility.

3.  **Model Training**:
    -   A **Logistic Regression** model is initialized.
    -   The model is trained on the training data (`X_train` and `y_train`) to learn the relationship between the features and the target variable.

4.  **Prediction and Evaluation**:
    -   The trained model predicts the diagnosis for the test data (`X_test`).
    -   The model's performance is evaluated using several metrics:
        -   **Accuracy Score**: Measures the overall correctness of the predictions.
        -   **Classification Report**: Provides a detailed breakdown of `precision`, `recall`, and `f1-score` for each class (`malignant` and `benign`).
        -   **Confusion Matrix**: A table showing the number of **true positives**, **true negatives**, **false positives**, and **false negatives**.

## ✅ Outcome

-   The model achieved a high **accuracy score of ~0.956**, meaning it correctly predicted the diagnosis for approximately 95.6% of the test samples.
-   The **classification report** shows strong performance for both classes, with high precision and recall.
    -   **Malignant (0)**: `Precision: 0.97`, `Recall: 0.91`
    -   **Benign (1)**: `Precision: 0.95`, `Recall: 0.99`
-   The **confusion matrix** visualization confirms the model's effectiveness:
    -   **True Positives (Benign)**: 70
    -   **True Negatives (Malignant)**: 39
    -   **False Positives**: 4 (predicts malignant, but is benign)
    -   **False Negatives**: 1 (predicts benign, but is malignant)

The results indicate that the **Logistic Regression** model is a strong and reliable classifier for this specific task, capable of accurately distinguishing between malignant and benign breast cancer cells based on the provided features.
