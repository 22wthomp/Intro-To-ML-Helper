
HELP_CONTENT = {
    # ============ MAIN INTERFACE COMPONENTS =====================
    
    "upload-data": {
        "tooltip": "Upload a CSV file with your dataset. First row should contain column headers.",
        "example": "Make sure your CSV has no missing values and all numeric columns for regression."
    },
    
    "target-col-input": {
        "tooltip": "The column you want to predict. Leave blank to use the last column automatically.",
        "example": "For house prices: 'price'. For email classification: 'spam_label'."
    },
    
    "split-ratio-dropdown": {
        "tooltip": "Percentage of data for training. Higher = more training data but less test data, 80/20 is common. Use 90/10 for small datasets, 70/30 for large ones.",
        "example": "80/20 is common. Use 90/10 for small datasets, 70/30 for large ones."
    },
    
    "include-val-checklist": {
        "tooltip": "Validation set helps detect overfitting during training. Recommended for hyperparameter tuning. Use validation when comparing many models or testing different parameters.",
        "example": "Use validation when comparing many models or testing different parameters."
    },
    
    "task-type": {
        "tooltip": "Classification predicts categories/labels, Regression predicts continuous numbers.",
        "example": "Classification: spam detection, image recognition. Regression: price prediction, temperature forecasting."
    },
    
    # ============== SCALERS =====================
    
    
    "scaler": {
        "tooltip": "Scaling options: Standard — centers data, good for linear models and SVMs. MinMax — compresses values to [0–1], best when features have fixed bounds. Robust — resistant to outliers, uses medians. MaxAbs — keeps sparsity and signs, good for sparse data. None — tree-based models (Random Forest, XGBoost) often don’t need scaling.",
    },
    
    # ============= CLASSIFICATION MODELS ================================
    
    "logreg": {
        "tooltip": "Logistic Regression: Linear model for classification. Fast, interpretable, probabilistic. Good baseline for binary classification. Works well with scaled features",
        "hyperparameters": {
            "C": {
                "tooltip": "Inverse regularization strength. Smaller values = more regularization. Try 0.01, 0.1, 1.0, 10, 100. Use smaller C to prevent overfitting.",
            },
            "max_iter": {
                "tooltip": "Maximum number of iterations for solver to converge. Default 1000 usually works. Increase if you get convergence warnings.",
            },
            "penalty": {
                "tooltip": "Type of regularization: 'l1' (Lasso), 'l2' (Ridge), or 'elasticnet'. 'l2' is default and most common. 'l1' for feature selection.",
            },
            "solver": {
                "tooltip": "Algorithm for optimization: 'liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'. 'liblinear' for small datasets, 'lbfgs' for multiclass, 'saga' for large datasets.",
            }
        }
    },
    
    "rf": {
        "tooltip": "Random Forest: Ensemble of decision trees. Handles mixed data, reduces overfitting.",
        "example": "Great general-purpose classifier. Works without scaling. Good for tabular data.",
        "hyperparameters": {
            "n_estimators": {
                "tooltip": "Number of decision trees in the forest. More trees = better performance but slower. Try 50, 100, 200, 500. Diminishing returns after ~200 for most datasets.",
            },
            "max_depth": {
                "tooltip": "Maximum depth of each tree. None = unlimited depth. Try 5, 10, 20, None. Use smaller values to prevent overfitting.",
            },
            "min_samples_split": {
                "tooltip": "Minimum samples required to split an internal node. Try 2, 5, 10. Higher values prevent overfitting on small datasets.",
            },
            "min_samples_leaf": {
                "tooltip": "Minimum samples required to be at a leaf node. Try 1, 2, 4. Higher values smooth the model and prevent overfitting.",
            },
            "max_features": {
                "tooltip": "Number of features to consider for best split: 'sqrt', 'log2', or integer. 'sqrt' is default for classification. 'log2' for high-dimensional data.",
            },
            "bootstrap": {
                "tooltip": "Whether to use bootstrap samples when building trees. True (default) uses bagging. False uses all data for each tree.",
            }
        }
    },
    
    "svc": {
        "tooltip": "Support Vector Classifier: Creates optimal decision boundary. Powerful for complex patterns.",
        "example": "Excellent for high-dimensional data like text. Requires feature scaling.",
        "hyperparameters": {
            "C": {
                "tooltip": "Regularization parameter. Higher C = less regularization, more complex model. Try 0.1, 1, 10, 100. Lower C for noisy data, higher C for clean data.",
            },
            "kernel": {
                "tooltip": "Kernel function: 'linear', 'rbf', 'poly', 'sigmoid'. 'rbf' (default) for non-linear data. 'linear' for high-dimensional sparse data.",
            },
            "gamma": {
                "tooltip": "Kernel coefficient for 'rbf', 'poly', 'sigmoid'. Higher = more complex boundaries. Try 'scale', 'auto', 0.001, 0.01, 0.1, 1. Higher gamma = more overfitting risk.",
            },
            "degree": {
                "tooltip": "Degree of polynomial kernel (only for 'poly' kernel). Try 2, 3, 4. Higher degrees create more complex decision boundaries.",
            }
        }
    },
    
    "knnc": {
        "tooltip": "K-Nearest Neighbors Classifier: Classifies based on majority vote of k nearest neighbors.",
        "example": "Simple, interpretable. Good for small datasets with clear clusters. Sensitive to scaling.",
        "hyperparameters": {
            "n_neighbors": {
                "tooltip": "Number of neighbors to consider. Odd numbers avoid ties in binary classification. Try 3, 5, 7, 11. Smaller k = more complex boundaries, larger k = smoother boundaries.",
            },
            "weights": {
                "tooltip": "Weight function: 'uniform' (all neighbors equal) or 'distance' (closer neighbors weighted more). 'uniform' for balanced data, 'distance' when closer neighbors are more relevant.",
            },
            "algorithm": {
                "tooltip": "Algorithm for nearest neighbor search: 'auto', 'ball_tree', 'kd_tree', 'brute'. 'auto' lets scikit-learn choose. 'ball_tree' for high dimensions, 'kd_tree' for low dimensions.",
            },
            "p": {
                "tooltip": "Power parameter for Minkowski distance: 1=Manhattan, 2=Euclidean. p=1 for Manhattan distance, p=2 (default) for Euclidean distance.",
            }
        }
    },
    
    "nb": {
        "tooltip": "Gaussian Naive Bayes: Assumes feature independence and normal distribution. Fast and simple.",
        "example": "Works well with small datasets and text classification. Good baseline model.",
        "hyperparameters": {
            "var_smoothing": {
                "tooltip": "Portion of largest variance added to all variances for numerical stability. Default 1e-9 usually works. Increase if you get numerical issues.",
            }
        }
    },
    
    "adac": {
        "tooltip": "AdaBoost Classifier: Combines weak learners (usually decision stumps) sequentially.",
        "example": "Good for binary classification. Can convert weak learners into strong classifier.",
        "hyperparameters": {
            "n_estimators": {
                "tooltip": "Maximum number of weak learners (estimators) to train. Try 50, 100, 200. More estimators = better performance but risk of overfitting.",
            },
            "learning_rate": {
                "tooltip": "Weight applied to each weak learner. Lower values require more estimators. Try 0.1, 0.5, 1.0, 2.0. Lower learning rate often gives better results.",
            },
            "algorithm": {
                "tooltip": "Boosting algorithm: 'SAMME' or 'SAMME.R'.'SAMME.R' (default) converges faster, 'SAMME' for base estimators without predict_proba.",
            }
        }
    },
    
    "xgbc": {
        "tooltip": "XGBoost Classifier: Advanced gradient boosting. Often wins ML competitions.",
        "example": "Excellent for structured/tabular data. Handles missing values, feature interactions.",
        "hyperparameters": {
            "n_estimators": {
                "tooltip": "Number of boosting rounds (trees to build). Try 100, 200, 500, 1000. More trees = better performance but longer training.",
            },
            "max_depth": {
                "tooltip": "Maximum depth of each tree. Controls model complexity. Try 3, 6, 10. Deeper trees can model complex interactions but may overfit.",
            },
            "learning_rate": {
                "tooltip": "Step size shrinkage to prevent overfitting. Also called 'eta'. Try 0.01, 0.1, 0.3. Lower values require more n_estimators but often perform better.",
            },
            "subsample": {
                "tooltip": "Fraction of samples used for each tree. Prevents overfitting. Try 0.8, 0.9, 1.0. Lower values add randomness and prevent overfitting.",
            },
            "colsample_bytree": {
                "tooltip": "Fraction of features used for each tree.Try 0.8, 0.9, 1.0. Lower values add randomness and speed up training.",
            },
            "reg_alpha": {
                "tooltip": "L1 regularization term on weights (Lasso). Try 0, 0.1, 1, 10. Higher values = more regularization.",
            },
            "reg_lambda": {
                "tooltip": "L2 regularization term on weights (Ridge). Try 0, 0.1, 1, 10. Higher values = more regularization.",
            }
        }
    },
    
    # ================= REGRESSION MODELS =========================
    
    "linreg": {
        "tooltip": "Linear Regression: Assumes linear relationship between features and target. Simple and interpretable.",
        "example": "Good baseline for regression. Works best when relationship is actually linear.",
        "hyperparameters": {
            "fit_intercept": {
                "tooltip": "Whether to fit an intercept term (y-axis crossing point). True (default) fits intercept. Set False if data is centered at origin.",
            },
            "normalize": {
                "tooltip": "Whether to normalize features before regression (deprecated, use scalers instead). Use StandardScaler or other scalers instead of this parameter.",
            }
        }
    },
    
    "ridge": {
        "tooltip": "Ridge Regression: Linear regression with L2 regularization. Prevents overfitting.",
        "example": "Use when you have many features or multicollinearity. Shrinks coefficients toward zero.",
        "hyperparameters": {
            "alpha": {
                "tooltip": "Regularization strength. Higher alpha = more regularization = simpler model. Try 0.1, 1.0, 10, 100. Cross-validation helps find optimal alpha.",
            },
            "fit_intercept": {
                "tooltip": "Whether to fit an intercept term. True (default) usually. Set False if data is pre-centered.",
            },
            "solver": {
                "tooltip": "Solver algorithm: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'.'auto' chooses best solver. 'sag'/'saga' for large datasets.",
            }
        }
    },
    
    "lasso": {
        "tooltip": "Lasso Regression: Linear regression with L1 regularization. Performs automatic feature selection.",
        "example": "Use for feature selection - sets irrelevant feature coefficients to exactly zero.",
        "hyperparameters": {
            "alpha": {
                "tooltip": "Regularization strength. Higher alpha = more features set to zero. Try 0.1, 1.0, 10. Higher alpha = sparser model with fewer features.",
            },
            "max_iter": {
                "tooltip": "Maximum number of iterations for coordinate descent solver. Default 1000. Increase if convergence warnings appear.",
            },
            "selection": {
                "tooltip": "Feature selection strategy: 'cyclic' or 'random'. 'cyclic' (default) updates features sequentially. 'random' can be faster for large datasets.",
            }
        }
    },
    
    "elastic": {
        "tooltip": "Elastic Net: Combines L1 (Lasso) and L2 (Ridge) regularization. Balances feature selection and shrinkage.",
        "example": "Good compromise between Ridge and Lasso. Handles correlated features better than Lasso.",
        "hyperparameters": {
            "alpha": {
                "tooltip": "Overall regularization strength. Higher alpha = more regularization. Try 0.1, 1.0, 10. Controls total amount of regularization.",
            },
            "l1_ratio": {
                "tooltip": "Mix of L1 vs L2: 0=Ridge only, 1=Lasso only, 0.5=equal mix. Try 0.1, 0.5, 0.7, 0.9. Higher values favor feature selection (L1).",
            },
            "max_iter": {
                "tooltip": "Maximum iterations for coordinate descent solver. Default 1000. Increase if you get convergence warnings.",
            }
        }
    },
    
    "rfr": {
        "tooltip": "Random Forest Regressor: Ensemble of decision trees for regression. Handles non-linear relationships.",
        "example": "Excellent general-purpose regressor. Works without scaling. Good for complex relationships.",
        "hyperparameters": {
            "n_estimators": {
                "tooltip": "Number of decision trees in the forest. Try 50, 100, 200, 500. More trees = better performance but slower training.",
            },
            "max_depth": {
                "tooltip": "Maximum depth of each tree. None = unlimited depth. Try 5, 10, 20, None. Limit depth to prevent overfitting.",
            },
            "min_samples_split": {
                "tooltip": "Minimum samples required to split an internal node. Try 2, 5, 10. Higher values prevent overfitting.",
            },
            "min_samples_leaf": {
                "tooltip": "Minimum samples required at leaf node. Try 1, 2, 4. Higher values create smoother predictions.",
            },
            "max_features": {
                "tooltip": "Number of features for best split: 'sqrt', 'log2', int, or float.'sqrt' is common. Try 'log2' or 1.0 (all features) for different randomness levels.",
            }
        }
    },
    
    "svr": {
        "tooltip": "Support Vector Regression: Uses support vectors to predict continuous values. Good for non-linear patterns.",
        "example": "Powerful for complex non-linear relationships. Requires feature scaling.",
        "hyperparameters": {
            "C": {
                "tooltip": "Regularization parameter. Higher C = less regularization. Try 0.1, 1, 10, 100. Higher C allows more complex models.",
            },
            "epsilon": {
                "tooltip": "Epsilon-tube width. Predictions within epsilon of true value have no penalty. Try 0.01, 0.1, 1.0. Larger epsilon = more tolerance for errors.",
            },
            "kernel": {
                "tooltip": "Kernel function: 'linear', 'rbf', 'poly', 'sigmoid'. 'rbf' (default) for non-linear data. 'linear' for high-dimensional data.",
            },
            "gamma": {
                "tooltip": "Kernel coefficient for 'rbf', 'poly', 'sigmoid'. Try 'scale', 'auto', 0.001, 0.01, 0.1. Higher gamma = more complex boundaries.",
            }
        }
    },
    
    "knnr": {
        "tooltip": "K-Nearest Neighbors Regressor: Predicts based on average of k nearest neighbors.",
        "example": "Simple, non-parametric. Good for local patterns. Sensitive to feature scaling.",
        "hyperparameters": {
            "n_neighbors": {
                "tooltip": "Number of neighbors to average for prediction. Try 3, 5, 7, 11. Smaller k = more local predictions, larger k = smoother predictions.",
            },
            "weights": {
                "tooltip": "Weight function: 'uniform' or 'distance'. 'uniform' treats all neighbors equally. 'distance' weights closer neighbors more.",
            },
            "algorithm": {
                "tooltip": "Algorithm for finding neighbors: 'auto', 'ball_tree', 'kd_tree', 'brute'. 'auto' lets scikit-learn choose optimal algorithm based on data.",
            }
        }
    },
    
    "xgbr": {
        "tooltip": "XGBoost Regressor: Advanced gradient boosting for regression. State-of-the-art for tabular data.",
        "example": "Often best performer for structured data competitions. Handles missing values and interactions.",
        "hyperparameters": {
            "n_estimators": {
                "tooltip": "Number of boosting rounds (trees). Try 100, 200, 500, 1000. More trees generally better but slower.",
            },
            "max_depth": {
                "tooltip": "Maximum depth of each tree. Try 3, 6, 10. Deeper trees capture interactions but may overfit.",
            },
            "learning_rate": {
                "tooltip": "Step size shrinkage (eta). Lower values need more n_estimators. Try 0.01, 0.1, 0.3. Lower learning rate often gives better final performance.",
            },
            "subsample": {
                "tooltip": "Fraction of training samples used for each tree. Try 0.8, 0.9, 1.0. Lower values add randomness and prevent overfitting.",
            },
            "colsample_bytree": {
                "tooltip": "Fraction of features used for each tree. Try 0.8, 0.9, 1.0. Lower values speed up training and add randomness.",
            },
            "reg_alpha": {
                "tooltip": "L1 regularization on leaf weights. Try 0, 0.1, 1. Higher values increase regularization.",
            },
            "reg_lambda": {
                "tooltip": "L2 regularization on leaf weights. Try 0, 0.1, 1. Higher values increase regularization.",
            }
        }
    },
    
    # ============ EVALUATION METRICS =======================================
    
    "metrics-accuracy": {
        "tooltip": "Percentage of correct predictions. Can be misleading with imbalanced classes. 90% accuracy means 9 out of 10 predictions correct.",
        "example": "90% accuracy means 9 out of 10 predictions correct. Use with balanced datasets."
    },
    
    "metrics-precision": {
        "tooltip": "Of positive predictions, how many were actually positive? TP/(TP+FP). High precision = few false alarms. Important when false positives are costly.",
        "example": "High precision = few false alarms. Important when false positives are costly."
    },
    
    "metrics-recall": {
        "tooltip": "Of actual positives, how many did we catch? TP/(TP+FN). High recall = catch most positive cases. Important when false negatives are costly.",
        "example": "High recall = catch most positive cases. Important when false negatives are costly."
    },
    
    "metrics-f1": {
        "tooltip": "Harmonic mean of precision and recall. Balances both metrics. Good single metric for imbalanced datasets. Range: 0 (worst) to 1 (best).",
        "example": "Good single metric for imbalanced datasets. Range: 0 (worst) to 1 (best)."
    },
    
    "metrics-auc": {
        "tooltip": "Area Under ROC Curve. Measures ability to distinguish between classes at all thresholds. 0.5 = random guessing, 1.0 = perfect classification.",
        "example": "0.5 = random guessing, 1.0 = perfect classification. Good for ranking problems."
    },
    
    "metrics-mse": {
        "tooltip": "Mean Squared Error. Average of squared differences. Penalizes large errors heavily. Lower is better. Units are squared target units (e.g., dollars²).",
        "example": "Lower is better. Units are squared target units (e.g., dollars²)."
    },
    
    "metrics-rmse": {
        "tooltip": "Root Mean Squared Error. Square root of MSE. Same units as target variable. More interpretable than MSE (e.g., in dollars not dollars²).",
        "example": "Lower is better. More interpretable than MSE (e.g., in dollars not dollars²)."
    },
    
    "metrics-mae": {
        "tooltip": "Mean Absolute Error. Average of absolute differences. Less sensitive to outliers than MSE. Lower is better. Same units as target. More robust to outliers than RMSE.",
        "example": "Lower is better. Same units as target. More robust to outliers than RMSE."
    },
    
    "metrics-r2": {
        "tooltip": "R² (coefficient of determination). Proportion of variance explained by model. 1.0 = perfect fit, 0.0 = as good as mean baseline, negative = worse than mean.",
        "example": "1.0 = perfect fit, 0.0 = as good as mean baseline, negative = worse than mean."
    },
    "metrics-max-error": {
        "tooltip": "Largest single absolute error between a predicted and actual value. Highlights the worst-case miss. If true = 100 and predicted = 60, the error is 40. This metric focuses on the largest mistake made by the model.",
        "example": "If true = 100 and predicted = 60, the error is 40. This metric focuses on the largest mistake made by the model."
    },
    "metrics-mape": {
        "tooltip": "Mean Absolute Percentage Error. Average of absolute percentage errors. 10% MAPE = average error of 10%.",
        "example": "10% MAPE = average error of 10%. Easy to interpret but undefined when actual = 0."
    },
    
    #==================GRAPHS====================================================================
    
    "confusion-matrix": {
        "tooltip": "Heatmap showing actual vs predicted classes. Perfect predictions appear on diagonal. Dark squares on diagonal = good predictions. Off-diagonal = confusion between classes.",
        "example": "Dark squares on diagonal = good predictions. Off-diagonal = confusion between classes."
    },
    
    "roc-curve": {
        "tooltip": "ROC Curve plots True Positive Rate vs False Positive Rate at different thresholds. Curve closer to top-left corner = better model. Diagonal line = random guessing.",
        "example": "Curve closer to top-left corner = better model. Diagonal line = random guessing."
    },
    
    "multiclass-roc": {
        "tooltip": "Multiple ROC curves showing each class vs all others (one-vs-rest approach). Each colored line represents one class. Higher curves = better discrimination for that class.",
        "example": "Each colored line represents one class. Higher curves = better discrimination for that class."
    },
    
    "feature-importance": {
        "tooltip": "Bar chart showing which features most influence model predictions. Longer bars = more important features. Help identify which variables drive your model.",
        "example": "Longer bars = more important features. Help identify which variables drive your model."
    },
    
    "predicted-vs-actual": {
        "tooltip": "Scatter plot comparing predicted values (y-axis) vs actual values (x-axis). Points closer to diagonal red line = better predictions. Spread indicates error magnitude.",
        "example": "Points closer to diagonal red line = better predictions. Spread indicates error magnitude."
    },
    
    "residuals-plot": {
        "tooltip": "Shows prediction errors (residuals) vs predicted values. Helps detect patterns in errors. Random scatter around zero = good model. Patterns indicate model bias or non-linearity.",
        "example": "Random scatter around zero = good model. Patterns indicate model bias or non-linearity."
    },
    
    "residuals-histogram": {
        "tooltip": "Distribution of prediction errors. Should be roughly normal (bell-shaped) for good models. Bell-shaped centered at zero = good. Skewed or multi-modal suggests model problems.",
        "example": "Bell-shaped centered at zero = good. Skewed or multi-modal suggests model problems."
    },
    
    "metrics-table": {
        "tooltip": "Comparison table showing performance metrics for test and validation sets. Similar test/validation scores = good generalization. Large differences = overfitting.",
        "example": "Similar test/validation scores = good generalization. Large differences = overfitting."
    },
}

def get_help_content(component_id):
    #Get help content for a component
    return HELP_CONTENT.get(component_id, {
        "tooltip": "No help available yet.",
        "example": ""
    })

def get_params(model_id):
    #get all params for appropriate model
    model_content = HELP_CONTENT.get(model_id, {})
    return model_content.get("hyperparameters",{})

def get_param_help(model_id,param):
    #get param content from specific model
    param_content = get_params(model_id)
    return param_content.get(param,{
        "tooltip": "No help available yet.",
        "example": ""
    })