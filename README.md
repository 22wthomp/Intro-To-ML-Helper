# Machine Learning Pipeline Dashboard
Interactive, beginner-friendly dashboard for exploring end-to-end machine learning pipelines without writing code.
# Overview
This project is a **Dash based web app** that lets you build, train, and evaluate machine learning models through an interactive dashboard.

It started as an education project for me to:
- Practice building ML Pipelines
- Applying topics I learned in class to an actual project
- Create a tool that helps people new to machine learning experimnt with different models, metrics and visualizations without touching code.
You upload a CSV, pick your target column, customize your preprocessing (split ratio, inclusion of validation set), choose Classification or Regression, select one or more models, optionally apply feature scaling or adjust parameters.
## Key Features
- **CSV Upload**
  - Upload a single CSV file with a header row
- **Target Column Selection**
  - Type the name of the target column
  - If left blank the dashboard automatically uses the last column in th edataset as the target.
-**Configurable Train/Test Split**
  - Choose the train/test ratio (70/30, 75/25, 80/20, 90/10)
  - Data is split into train and test sets using sckit-learn's train_test_split
-**Optional Validation Set**
  - Check a box to create a validation set (20% of the training portion)
- **Classification or Regression Mode**
  - Switch between
    - Classification
    - Regression
  -The list of available models updates automatically based on the chosen task
- **Multiple Models in One Run**
  - Select one or more models at the same time via a checklist
  - The app trains and evaluates each selected model and shows results one after another
- **Per-Model Feature Scaling**
  - Each model has its own scaler dropdown:
    - None
    - StandardScaler
    - MinMaxScaler
    - RobustScaler
    - MaxAbsScaler
  - Scaling is applied only to the models you choose, so you can see how scaling affects performance
- **Interactive Hyperparameter Controls**
  - For each model, a parameter panel appears with the most important hyperparameters:
      - Numeric fields (C, alpha, n_estimators, max_depth, etc)
      - Dropdown for choices(e.g., kernels, eval metics)
      - Checkboxes for boolean options
    - Each parameter has an ℹ️ help icon that explains what it does and how you would want to change it
- **Automatic Metrics & Visualizations**
  - The app computes appropriate metrics tables and plots for every model:
    - Classification: accuracy, precision, recall, F1, AUC + ROC curves, confusion matrices, multiclass ROC, feature importance
    - Regression: MSE, RMSE, MAE, R², Max Error, MAPE + predicted vs actual, residual plots, residual histograms.
  - All plots are rendered with Plotly inside the dashboard (and therefore can be adjusted for better visualization).
- **Educational Tooltips**
  - ℹ️ icons next to metrics, graphs, scalers,hyperparameters and models show friendly explanations and examples (e.g., what AUC means, how MAPE is interpreted, why scaling matters).
## Supported Models & Metrics
**Classification**
Classification Models (via scikit-learn & XGBoost):
  - Logistic Regression (logreg)
  - Random Forest Classifier (rf)
  - Support Vector Classifier (svc)
  - K-Nearest Neighbors Classifier (knnc)
  - Gaussian Naive Bayes (nb)
  - AdaBoost Classifier (adac)
  - XGBoost Classifier (xgbc)
**Classification Metrics**
For each model, the app computes metrics on test (and optionally validation) data:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - AUC (Area Under the ROC Curve)
These metrics are displayed in a table with rows for Test and Validation and each metric has an explanatory tooltip
**Classification Plots**
  - Confusion Matrix Heatmap
    - Shows how often each class was correctly or incorrectly predicted
    - Dark diagonal = good; off-diagonal cells show where the model is confused.
  - ROC Curve
    - Plots True Positive Rate vs False Positive Rate at different thresholds.
    - Helps visualize the trade-off between sensitivity and specificity for binary problems
  - Multicalss ROC
    - One-vs-rest ROC curves for each class, useful to see which classes are easier or harder to distinguish
  - Feature Importance Bar Chart
