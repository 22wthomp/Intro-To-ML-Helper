
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State,ALL
import base64
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,label_binarize, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, AdaBoostClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score,roc_curve, confusion_matrix,mean_squared_error, mean_absolute_error, r2_score, max_error
from xgboost import XGBClassifier, XGBRegressor

from help_text import get_help_content,get_params,get_param_help, HELP_CONTENT


def create_help_icon(component_id, text=""):
    # give it a real id so Tooltips/Popovers can target it
    help_content = get_help_content(component_id)
    return html.Span(
        "ℹ️",
        id=f"{component_id}-help",
        title=help_content["tooltip"] or "More info",  # keeps fallback browser tooltip too
        style={"marginLeft": "6px", "cursor": "help", "color": "#0d6efd",
               "display": "inline-block"}
    )
def create_param_help(model, param):
    # given a model and hyperparam it will then return the tool tip for that
    help_content = get_param_help(model,param)
    return html.Span(
        "ℹ️",
        id=f"{model}-{param}-help",
        title=help_content.get("tooltip") or "More info",  # keeps fallback browser tooltip too
        style={"marginLeft": "6px", "cursor": "help", "color": "#0d6efd",
               "display": "inline-block"}
    )
def create_scaler_section_header():
    #creates a header for the scaler section with general help about scaling.
    return html.Div([
            "Feature Scaling Options",
            create_help_icon("scaler")
    ])

def create_Scaler_Dropdown(model_id,model_name,default_scaler):
    #creates scaler drop down for each model selected
    return html.Div(
        id=f"scaler-{model_id}-container",
        style={"display": "none", "marginBottom": "20px"},
        children=[
            html.Label(f"Scaler for {model_name}:"),
            dcc.Dropdown(
                id=f"scaler-{model_id}",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "StandardScaler", "value": "standard"},
                    {"label": "MinMaxScaler", "value": "minmax"},
                    {"label": "RobustScaler", "value": "robust"},
                    {"label": "MaxAbsScaler", "value": "maxabs"},
                ],
                value=default_scaler,
                clearable=False,
                style={"width": "200px"},
            ),
        ],
    )

SENTINEL_NONE = "__NONE__"
def _encode_none(v):
    return SENTINEL_NONE if v is None else v

def _decode_none(default, v):
    if v == SENTINEL_NONE and default is None:
        return None
    return v
def _number_step(val):
    #sets how much each number option should go up by
    if isinstance(val, int): return 1 if abs(val) < 100 else 10
    if isinstance(val, float): return 0.1 if abs(val) >= 0.1 else 0.01
    return 1

def _coerce_bool(default, value):
    # Checklist returns [] or [True]; keep other types unchanged
    if isinstance(default, bool):
        return True if (isinstance(value, list) and True in value) else False
    return value

def _control_for(model: str, name: str, default):
    #creates the appropriate parameter option according to the type of default it is
    cid = {"type": "param", "model": model, "name": name}

    # creates contrl for booleans -> checkbox
    if isinstance(default, bool):
        return dcc.Checklist(
            id=cid,
            options=[{"label": "", "value": True}],
            value=[True] if default else [],
            style={"display": "inline-block", "marginLeft": "8px"}
        )

    # creates control for categorical option uses predefined choices from PARAM-CHOICES dict (also supports None)
    choices = PARAM_CHOICES.get(model, {}).get(name)
    if choices is not None:
        opts = []
        if any(c is None for c in choices):
            opts.append({"label": "None", "value": SENTINEL_NONE}) #SENTINEL_NONE is "__NONE__" dash automatically fills unfilled values with None so we use this as our placeholder
        opts += [{"label": str(x), "value": x} for x in choices if x is not None]

        return dcc.Dropdown(
            id=cid,
            options=opts,
            value=_encode_none(default), #encodes any None values found with sentinel None
            clearable=False,
            style={"width": 180, "display": "inline-block", "marginLeft": "8px"}
            )

    #creates control for numbers input
    if isinstance(default, (int, float)):
        return dcc.Input(
            id=cid, type="number", value=default,
            step=_number_step(default), debounce=True,
            **({"min": 0} if isinstance(default, int) else {}),
            style={"width": 160, "display": "inline-block", "marginLeft": "8px"}
        )

    # free text for the rare chance we get this
    return dcc.Input(
        id=cid, type="text", value=str(default),
        style={"width": 200, "display": "inline-block", "marginLeft": "8px"}
    )

def build_param_section(model: str):
    #builds parameter options
    rows = []
    for pname, pdefault in PARAM_DEFAULTS[model].items():
        control = _control_for(model, pname, pdefault) #creates each parameter option
        rows.append(
            html.Div(
                [
                    html.Label(pname, style={"minWidth": 140}),
                    create_param_help(model, pname),   # ← separate child
                    control
                ],
                style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "8px"}
            )
        )
    return html.Fieldset(
        [html.Legend(f"{MODEL_DEFINITIONS[model]['name']} — Parameters", style={"fontWeight": 600})] + rows,
        style={"border": "1px solid #ccc", "borderRadius": "8px", "padding": "10px", "marginBottom": "12px"}
    )
def validate_clean_dataset(df,task_type):
    
    #Validates that the dataset is clean (no missing values).
    #Returns (is_valid, error_message)
    
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]
    #looks for non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    errors = []
    if len(missing_cols) > 0:
        errors.append(f"Missing values in: {list(missing_cols.index)}")
    if len(non_numeric_cols) > 0 and task_type == "reg":
        errors.append(f"Non-numeric columns: {non_numeric_cols}")
    
    if errors:
        return False, "Dataset must be clean with all numeric columns. Issues found: " + "; ".join(errors)
    
    return True, None

def fill_info_lines(df,target_col,include_val,X_train,X_test,X_val,info_lines):
    if target_col and (target_col in df.columns):
        info_lines.append(html.Div(f"Using '{target_col}' as target column."))
    else:
        info_lines.append(
            html.Div(
                f"No valid target entered so using the last column: '{df.columns[-1]}' as target. "
            )
        )
    # Show the sizes of train/test (and val if any)
    info_lines.append(
        html.Div(
            f"Train rows: {X_train.shape[0]}, "
            + (f"Val rows: {X_val.shape[0]}, " if include_val else "")
            + f"Test rows: {X_test.shape[0]}"
        )
    )
    return None

def split_data(df,target_col,split_ratio,include_val):
    #Splits df into X_train, X_test (and optionally X_val) plus y_train, y_test (and y_val).
    #Returns a dict with keys:X_train, X_val (or None), X_test, y_train, y_val (or None), y_test.
    
    random_state = np.random.randint(0, 2**16)
    #finds the “target” column
    if target_col and (target_col in df.columns):
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        # default: use the last column
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    # First split: train_val vs. test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=1 - split_ratio, random_state=random_state
    )

    if include_val:
        # 3) Further split train_val into train + val (20% of train_val goes to val)
        #    i.e. val_size = 0.2 * (train_val size)
        #    since train_val size = split_ratio * total, this ends up being 0.2 * split_ratio of total.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.2, random_state=random_state
        )
    else:
        # no validation set; mark X_val/y_val as None
        X_train, X_val, y_train, y_val = X_train_val, None, y_train_val, None

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

def getErrorMetrics(y_test,y_pred,y_prob):

    acc = accuracy_score(y_test, y_pred)
    classes = np.unique(y_test)

    if len(classes) == 2:
        # if there are only 2 classes use average='binary'
        prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    else:
        # if there are more than 2 classes use average='macro'
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # 3) AUC
    #    - If binary, y_prob has shape (n,2) and we take column 1.
    #    - If multiclass, do one-vs-rest with average="macro".
    y_prob_arr = np.array(y_prob)
    if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
        # Binary: take probability of “positive” (column 1)
        pos_proba = y_prob_arr[:, 1]
        auc = roc_auc_score(y_test, pos_proba)
    else:
        # Multiclass: label-binarize and one-vs-rest
        y_true_bin = label_binarize(y_test, classes=classes)
        auc = roc_auc_score(y_true_bin, y_prob_arr, average="macro", multi_class="ovr")

    conf_mat = confusion_matrix(y_test, y_pred)

    return acc, prec , recall, f1, auc, conf_mat

def getRegErrorMetrics(y_test, y_pred):
    # Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    # Root MSE
    rmse = np.sqrt(mse)
    # Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    # R2 Score
    r2 = r2_score(y_test, y_pred)
    
    max_err = max_error(y_test, y_pred)
    
    mask = y_test != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    else:
        mape = np.inf

    return mse, rmse, mae, r2, max_err, mape





def create_ROC_Curve(y_test, y_proba, title="ROC Curve"):
    #creates ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig_roc = px.line(
        x=fpr, y=tpr,
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
        title=title,
    )
    fig_roc.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,line=dict(dash="dash", color="gray"))

    return dcc.Graph(figure=fig_roc)

def create_multiclass_ROC(y_test, y_proba, model,classes, title="Multiclass ROC"):
    """
    Returns a dcc.Graph of one-vs-rest ROC curves for each class.
    - y_test:  true labels (array-like, length n_samples)
    - y_proba: model.predict_proba(X_test), shape (n_samples, n_classes)
    - classes: list or array of class labels in the same order as y_proba’s columns
    """
    #  Use the model’s class ordering
    #classes = model.classes_
    # 1) Binarize true labels
    
    y_test_bin = label_binarize(y_test, classes=classes)

    # 2) Compute FPR/TPR for every class
    fpr_dict = {}
    tpr_dict = {}
    for idx, cls in enumerate(classes):
        fpr_i, tpr_i, _ = roc_curve(y_test_bin[:, idx], y_proba[:, idx])
        fpr_dict[cls] = fpr_i
        tpr_dict[cls] = tpr_i

    # 3) Build the Plotly figure
    fig = go.Figure()
    for cls in classes:
        fig.add_trace(
            go.Scatter(
                x=fpr_dict[cls],
                y=tpr_dict[cls],
                mode="lines",
                name=f"{cls} vs Rest",
            )
        )

    # 4) Add diagonal baseline
    fig.add_shape(
        type="line",
        x0=0, x1=1, y0=0, y1=1,
        line=dict(dash="dash", color="gray"),
    )

    fig.update_layout(
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="Class",
        width=600,
        height=500,
    )
    return dcc.Graph(figure=fig)

def create_metrics_table(test_metrics,val_metrics):
    acc_test = test_metrics["accuracy"]
    prec_test = test_metrics["precision"]
    recall_test = test_metrics["recall"]
    f1_test = test_metrics["f1"]
    auc_test = test_metrics["auc"]

    cell_style = {
        "padding": "8px 12px",     # 8px vertical, 12px horizontal
        "border": "1px solid #ccc", # match existing border
        "textAlign": "center",
    }

    table_header = html.Tr(
        [
            html.Th("Dataset"),
            html.Th(["Accuracy",create_help_icon("metrics-accuracy")],style=cell_style),
            html.Th(["Precision",create_help_icon("metrics-precision")],style=cell_style),
            html.Th(["Recall",create_help_icon("metrics-recall")],style=cell_style),
            html.Th(["F1",create_help_icon("metrics-f1")],style=cell_style),
            html.Th(["AUC",create_help_icon("metrics-auc")],style=cell_style)
        ]
    )
    table_rows = [
        html.Tr([
            html.Td("Test",style=cell_style),
            html.Td(f"{acc_test:.2f}",style=cell_style),
            html.Td(f"{prec_test:.2f}",style=cell_style),
            html.Td(f"{recall_test:.2f}",style=cell_style),
            html.Td(f"{f1_test:.2f}",style=cell_style),
            html.Td(f"{auc_test:.2f}",style=cell_style)
            ])
    ]
    if val_metrics is not None:
        acc_val = val_metrics["accuracy"]
        prec_val = val_metrics["precision"]
        recall_val = val_metrics["recall"]
        f1_val = val_metrics["f1"]
        auc_val = val_metrics["auc"]
        table_rows.append(
            html.Tr([
                html.Td("Validation",style=cell_style),
                html.Td(f"{acc_val:.2f}",style=cell_style),
                html.Td(f"{prec_val:.2f}",style=cell_style),
                html.Td(f"{recall_val:.2f}",style=cell_style),
                html.Td(f"{f1_val:.2f}",style=cell_style),
                html.Td(f"{auc_val:.2f}",style=cell_style)
                ])
        )
    else:
        table_rows.append(
            html.Tr([
                html.Td("Validation",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style)
                ])
        )
    return html.Table([table_header] + table_rows, style={"border": "1px solid #ccc", "borderCollapse": "collapse"})

def create_metrics_table_reg(test_metrics,val_metrics):
    #retrieving test metrics
    mse_test = test_metrics["mse"]
    rmse_test = test_metrics["rmse"]
    mae_test = test_metrics["mae"]
    r2_test = test_metrics["r2"]
    max_error_test = test_metrics["max_error"]
    mape_test = test_metrics["mape"]
    
    cell_style = {
        "padding": "8px 12px",     # 8px vertical, 12px horizontal
        "border": "1px solid #ccc", # match existing border
        "textAlign": "center",
    }
    table_header = html.Tr(
        [
            html.Th("Dataset"), 
            html.Th(["MSE",create_help_icon("metrics-mse")],style=cell_style),
            html.Th(["RMSE",create_help_icon("metrics-rmse")],style=cell_style),
            html.Th(["MAE",create_help_icon("metrics-mae")],style=cell_style),
            html.Th(["R2",create_help_icon("metrics-r2")],style=cell_style),
            html.Th(["Max Error",create_help_icon("metrics-max-error")],style=cell_style),
            html.Th(["MAPE",create_help_icon("metrics-mape")],style=cell_style),
        ]
    )
    table_rows = [
        html.Tr([
            html.Td("Test",style=cell_style),
            html.Td(f"{mse_test:.2f}",style=cell_style),
            html.Td(f"{rmse_test:.2f}",style=cell_style),
            html.Td(f"{mae_test:.2f}",style=cell_style),
            html.Td(f"{r2_test:.2f}",style=cell_style),
            html.Td(f"{max_error_test:.2f}",style=cell_style),
            html.Td(f"{mape_test:.2f}",style=cell_style)
            ])
    ]
    if val_metrics is not None:
        mse_val = val_metrics["mse"]
        rmse_val = val_metrics["rmse"]
        mae_val = val_metrics["mae"]
        r2_val = val_metrics["r2"]
        max_error_val = val_metrics["max_error"]
        mape_val = val_metrics["mape"]
        
        table_rows.append(
            html.Tr([
                html.Td("Validation",style=cell_style),
                html.Td(f"{mse_val:.2f}",style=cell_style),
                html.Td(f"{rmse_val:.2f}",style=cell_style),
                html.Td(f"{mae_val:.2f}",style=cell_style),
                html.Td(f"{r2_val:.2f}",style=cell_style),
                html.Td(f"{max_error_val:.2f}",style=cell_style),
                html.Td(f"{mape_val:.2f}",style=cell_style)
                ])
        )
    else: #if there is no validation placeholders will be put
        table_rows.append(
            html.Tr([
                html.Td("Validation",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                html.Td("-",style=cell_style),
                ])
        )
    return html.Table([table_header] + table_rows, style={"border": "1px solid #ccc", "borderCollapse": "collapse"})
def create_confusion_matrix_graph(conf_mat, class_labels, title="Confusion Matrix"):
    """
    Returns a dcc.Graph of a heatmap for the confusion matrix.
    - conf_mat: 2D NumPy array from sklearn.metrics.confusion_matrix
    - class_labels: list/array of class names in the same order used when computing conf_mat
    """
    # Turn into a DataFrame so px.imshow shows row/column labels
    df_cm = pd.DataFrame(conf_mat, index=class_labels, columns=class_labels)

    fig = px.imshow(
        df_cm,
        text_auto=True,
        color_continuous_scale="Oranges",
        labels={"x": "Predicted Label", "y": "True Label"},
        title=title,
    )
    fig.update_layout(xaxis={"side": "top"})  # put x‐axis labels on top
    return dcc.Graph(figure=fig)

def create_feature_importance_bar(model, feature_names, title="Feature Importance"):
    """
    Returns a dcc.Graph that shows a bar chart of feature importances.
    - For LogisticRegression: uses abs(coef_[0])
    - For RandomForest: uses feature_importances_
    - feature_names: list or Index of column names (in the same order)
    """
    # 1) Extract importances depending on model type
    if hasattr(model, "coef_"):
        # LogisticRegression or other linear models
        coefs = model.coef_[0]  # shape (n_features,)
        importances = np.abs(coefs)
    elif hasattr(model, "feature_importances_"):
        # RandomForest or other tree‐based models
        importances = model.feature_importances_
    else:
        # If the estimator has no standard “feature importance,” return None
        return None

    # 2) Build a DataFrame for plotting
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # 3) Create a horizontal bar chart (so long names are easier to read)
    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
        labels={"importance": "Importance", "feature": "Feature"},
        color_discrete_sequence=["#00798C"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    return dcc.Graph(figure=fig)

def create_predicted_vs_actual_plot(y_actual, y_predicted, title="Predicted vs Actual"):
    """
    Creates a scatter plot comparing predicted vs actual values for regression.
    - y_actual: true target values
    - y_predicted: model predictions
    """
    # Create the scatter plot
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=y_actual,
            y=y_predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color='blue',
                size=6,
                opacity=0.6
            )
        )
    )
    
    # Add perfect prediction line (diagonal)
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        width=600,
        height=500,
        showlegend=True
    )
    
    return dcc.Graph(figure=fig)

def create_residuals_plot(y_actual, y_predicted, title="Residuals Plot"):
    """
    Creates a residuals plot showing prediction errors vs predicted values.
    - y_actual: true target values
    - y_predicted: model predictions
    """
    # Calculate residuals (actual - predicted)
    residuals = y_actual - y_predicted
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(
        go.Scatter(
            x=y_predicted,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(
                color='blue',
                size=6,
                opacity=0.6
            )
        )
    )
    
    # Add horizontal line at y=0
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Perfect Prediction"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        width=600,
        height=500,
        showlegend=False
    )
    
    return dcc.Graph(figure=fig)

def create_residuals_histogram(y_actual, y_predicted, title="Residuals Distribution"):
    """
    Creates a histogram of residuals to check for normal distribution.
    - y_actual: true target values
    - y_predicted: model predictions
    """
    # Calculate residuals
    residuals = y_actual - y_predicted
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=20,
            name='Residuals',
            marker_color='skyblue',
            opacity=0.7
        )
    )
    
    # Add vertical line at x=0
    fig.add_vline(
        x=0, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Zero Error"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Residuals (Actual - Predicted)",
        yaxis_title="Frequency",
        width=600,
        height=500,
        showlegend=False
    )
    
    return dcc.Graph(figure=fig)

def applyScaling(X_train, X_test, X_val, chosenScaler):
    scaler = None
    if chosenScaler in ["standard","minmax","robust","maxabs"]:
        scalerMap = {
            "standard":StandardScaler(),
            "minmax":MinMaxScaler(),
            "robust":RobustScaler(),
            "maxabs":MaxAbsScaler()
            }
        scaler = scalerMap[chosenScaler]
        Xtrain = scaler.fit_transform(X_train)
        Xtest = scaler.transform(X_test)
        if X_val is not None:
            Xval = scaler.transform(X_val)
        else:
            Xval = None
    else:
        Xtrain, Xtest = X_train, X_test
        if X_val is not None:
            Xval = X_val
        else:
            Xval = None
    return Xtrain, Xtest, Xval, scaler

def modelSelector(modelName,params):
    #Temp model selector future improvements will let the user specify parameters
    if params == None:
        params = {
            "logreg": {"max_iter" : 1000, "random_state":42},
            "rf": {  "n_estimators":100, "max_depth":None, "random_state":42},
            "svc":{"probability": True, "random_state": 42, "gamma": 'scale'},
            "knnc":{"n_neighbors":5},
            "nb":{"var_smoothing": 1e-9},
            "adac":{"n_estimators":50, "random_state":42},
            "xgbc":{"eval_metric":'logloss', "random_state":42},
            #Regression Models
            "linreg": {},
            "lasso": {"alpha": 1.0, "max_iter": 1000, "random_state": 42},
            "ridge": {"alpha": 1.0, "random_state": 42},
            "elastic": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000, "random_state": 42},
            "rfr": {"n_estimators": 100, "max_depth": None, "random_state": 42},
            "svr": {"kernel": 'rbf', "C": 1.0, "gamma": 'scale', "epsilon": 0.1},
            "knnr": {"n_neighbors": 5, "weights": 'uniform', "algorithm": 'auto'},
            "xgbr": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
        }
    if modelName == "logreg":
        model = LogisticRegression(**params["logreg"])
    elif modelName == "rf":
        model = RandomForestClassifier(**params["rf"])
    elif modelName == "svc":
        model = SVC(**params["svc"])
    elif modelName == "knnc":
        model = KNeighborsClassifier(**params["knnc"])
    elif modelName == "nb":
        model = GaussianNB(**params["nb"])
    elif modelName == "adac":
        model = AdaBoostClassifier(**params["adac"])
    elif modelName == "xgbc":
        model = XGBClassifier(**params["xgbc"])
    
    #Regression Models
    elif modelName == "linreg":
        model = LinearRegression(**params["linreg"])
    elif modelName == "lasso":
        model = Lasso(**params["lasso"])
    elif modelName == "ridge":
        model = Ridge(**params["ridge"])
    elif modelName == "elastic":
        return ElasticNet(**params["elastic"])
    elif modelName == "rfr":
        return RandomForestRegressor(**params["rfr"])
    elif modelName == "svr":
        return SVR(**params["svr"])
    elif modelName == "knnr":
        return KNeighborsRegressor(**params["knnr"])
    elif modelName == "xgbr":
        return XGBRegressor(**params["xgbr"])
    return model

def trainClassification(X_train,X_test,X_val,y_train,y_test,y_val,chosenScaler,modelName, params):
    Xtrain, Xtest, Xval, scaler = applyScaling(X_train, X_test, X_val, chosenScaler)
    le = None
    #apply logreg here
    if modelName == "xgbc":
        # XGB classifier expects classes to be number so we Encode labels ---
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)    # maps classes to 0,1,2
        y_test = le.transform(y_test)         # uses same mapping
        if y_val is not None:
            y_val = le.transform(y_val)
        else:
            y_val = None
    model = modelSelector(modelName,params)

    model.fit(Xtrain,y_train)

    y_pred = model.predict(Xtest)
    y_proba = model.predict_proba(Xtest)
    acc_test, prec_test, recall_test, f1_test, auc_test, conf_mat_test = getErrorMetrics(y_test,y_pred,y_proba)
    test_metrics = {
        "accuracy":acc_test,
        "precision":prec_test,
        "recall":recall_test,
        "f1":f1_test,
        "auc":auc_test,
    }
    if X_val is not None:
        y_pred_val = model.predict(Xval)
        y_proba_val = model.predict_proba(Xval)
        acc_val, prec_val, recall_val, f1_val, auc_val, conf_mat_val = getErrorMetrics(y_val,y_pred_val,y_proba_val)
        val_metrics = {
            "accuracy":acc_val,
            "precision":prec_val,
            "recall":recall_val,
            "f1":f1_val,
            "auc":auc_val,
        }
    else:
        #Default val_metrics to None
        val_metrics = None
        y_proba_val = None
        conf_mat_val = None
    #print("MODEL:", modelName, type(model).__name__)
    #print("PRED SUMMARY:", np.unique(y_pred, return_counts=True))

    return {
        "model": model,
        "scaler": scaler,  # Add this
        "label_encoder": le,  # Add this
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "y_test_proba": y_proba,
        "y_val_proba": y_proba_val,
        "conf_mat_test": conf_mat_test,
        "conf_mat_val": conf_mat_val,
    }
def trainRegression(X_train, X_test, X_val, y_train, y_test, y_val, chosenScaler, modelName, params):
    #trying to make one function that trains all Regression models, in future do this with the Classification models
    Xtrain, Xtest, Xval, scaler = applyScaling(X_train, X_test, X_val, chosenScaler)
    model = modelSelector(modelName, params)
    model.fit(Xtrain, y_train)
    
    # Regression predictions
    y_pred = model.predict(Xtest)
    
    # Regression metrics
    mse, rmse, mae, r2, max_error, mape = getRegErrorMetrics(y_test, y_pred)
    test_metrics = {
        "mse":mse,
        "rmse":rmse,
        "mae":mae,
        "r2":r2,
        "max_error": max_error,
        "mape": mape,
        }
    if X_val is not None:
        y_pred_val = model.predict(Xval)
        val_mse, val_rmse, val_mae, val_r2, val_max_error, val_mape= getRegErrorMetrics(y_val, y_pred_val)
        
        val_metrics = {
            "mse": val_mse,
            "rmse": val_rmse,
            "mae": val_mae, 
            "r2": val_r2,
            "max_error": val_max_error,
            "mape": val_mape
        }
    else:
        val_metrics = None
        y_pred_val = None
    return {
        "model": model,
        "scaler": scaler,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "y_test_pred": y_pred,  # No probabilities
        "y_val_pred": y_pred_val,
    }

def output_collector(modelName,model_out,y_test,X_train,classes,results):
    #----------
    #function collects output and applies graphing/table functions
    #--------
    test_metrics = model_out["test_metrics"]
    # Might be None if no validation set:
    val_metrics = model_out["val_metrics"]
    #Metric tables
    # Get the label encoder if it was used
    label_encoder = model_out["label_encoder"]
    if label_encoder is not None:
        y_test_original = label_encoder.inverse_transform(y_test)
        conf_matrix_labels = classes 
    else:
        y_test_original = y_test
        conf_matrix_labels = model_out["model"].classes_
    y_test_for_roc = y_test_original
    results.append(
        html.Div(
            [
                html.H1(modelName),
                create_metrics_table(test_metrics,val_metrics),
            ],
                style={"marginBottom": "20px"},
                )
            )
    #Confusion Matrix heat maps
    results.append(
        html.Div([
            html.Div([
                html.H4("Confusion Matrix (Test)", style={"display": "inline-block", "marginRight": "6px"}),
                create_help_icon("confusion-matrix")],
                style={"display": "flex", "alignItems": "center"}),
            create_confusion_matrix_graph(
                model_out["conf_mat_test"],
                conf_matrix_labels, #.classes_ makes sure that the order of the classes will remain the same
                title=""
            )
        ])
    )

    # If there is a validation confusion matrix, show that too
    if model_out["conf_mat_val"] is not None:
        results.append(
            html.Div([
                html.Div([
                    html.H4("Confusion Matrix (Validation)", style={"display": "inline-block", "marginRight": "6px"}),
                    create_help_icon("confusion-matrix")],
                    style={"display": "flex", "alignItems": "center"}),
                create_confusion_matrix_graph(
                    model_out["conf_mat_val"],
                    conf_matrix_labels,
                    title=""
                )
            ])
        )
    #ROC curves
    if len(classes) == 2:
        results.append(
            html.Div([
                html.Div([
                    html.H4("ROC Curve", style={"display": "inline-block", "marginRight": "6px"}),
                    create_help_icon("roc-curve")],
                    style={"display": "flex", "alignItems": "center"}),
                create_ROC_Curve(
                    y_test_for_roc,
                    model_out["y_test_proba"],
                    title=""
                )
            ])
        )
    else:
        results.append(
            html.Div([
                html.Div([
                    html.H4("Multiclass ROC Curve", style={"display": "inline-block", "marginRight": "6px"}),
                    create_help_icon("multiclass-roc")],
                    style={"display": "flex", "alignItems": "center"}),
                create_multiclass_ROC(
                    y_test_for_roc,
                    model_out["y_test_proba"],
                    model_out["model"],
                    classes,
                    title=""
                )
            ])
        )
        
    #feature importance bar graph
    results.append(
        html.Div([
            html.Div([
                html.H4("Feature Importance", style={"display": "inline-block", "marginRight": "6px"}),
                create_help_icon("feature-importance")],
                style={"display": "flex", "alignItems": "center"}),
            create_feature_importance_bar(
                model_out["model"],
                X_train.columns,
                title=""
            )
        ])
    )
    results.append(html.Hr(style={"border": "2px solid #444","margin": "20px 0"}))
    return None
def output_collector_regression(modelName, model_out, y_test, X_train, results):
    test_metrics = model_out["test_metrics"]
    # Might be None if no validation set:
    val_metrics = model_out["val_metrics"]
    #creates metric table
    results.append(
        html.Div(
            [
                html.H1(modelName),
                create_metrics_table_reg(test_metrics,val_metrics),
            ],
                style={"marginBottom": "20px"},
                )
            )
    #creates predicted vs actual plot
    results.append(
        html.Div([
            html.Div([
                html.H4(f"{modelName} - Predicted vs Actual", style={"display": "inline-block", "marginRight": "6px"}),
                create_help_icon("predicted-vs-actual")],
                style={"display": "flex", "alignItems": "center"}),
            create_predicted_vs_actual_plot(
                y_test, 
                model_out["y_test_pred"], 
                title=""
            )
        ])
    )
    #creates residuals plot
    results.append(
        html.Div([
            html.Div([
                html.H4(f"{modelName} - Residuals Plot", style={"display": "inline-block", "marginRight": "6px"}),
                create_help_icon("residuals-plot")],
                style={"display": "flex", "alignItems": "center"}),
            create_residuals_plot(
                y_test, 
                model_out["y_test_pred"], 
                title=""
            )
        ])
    )
    #creates residuals histogram
    results.append(
        html.Div([
            html.Div([
                html.H4(f"{modelName} - Residuals Distribution", style={"display": "inline-block", "marginRight": "6px"}),
                create_help_icon("residuals-histogram")],
                style={"display": "flex", "alignItems": "center"}),
            create_residuals_histogram(
                y_test, 
                model_out["y_test_pred"], 
                title=""
            )
        ])
    )
    return None

def read_results(test_metrics, val_metrics):
    infrences = []
    
    return infrences
def create_control_param_tune(model_id ):
    return dcc.Checklist(
        id="{model_id}-tune",
        options=[{"label": "", "value": True}],
        value=[],
        style={"display": "inline-block", "marginLeft": "8px"}
    )

def tune_models(model,params):
    return None
#Start of the Dash App
app = dash.Dash(__name__)
server = app.server  # for deployment later
#Model definition: use this dict for adding future models
MODEL_DEFINITIONS = {
    # Classification models
    "logreg": {"name": "Logistic Regression", "type": "clf", "default_scaler": "standard"},
    "rf": {"name": "Random Forest Classifier", "type": "clf", "default_scaler": "none"}, 
    "svc": {"name": "SVC", "type": "clf", "default_scaler": "standard"},
    "knnc": {"name": "KNeighbors Classifier", "type": "clf", "default_scaler": "standard"},
    "nb": {"name": "Gaussian Naive Bayes", "type": "clf", "default_scaler": "none"},
    "adac": {"name": "AdaBoost Classifier", "type": "clf", "default_scaler": "none"},
    "xgbc": {"name": "XGBoost Classifier", "type": "clf", "default_scaler": "none"},
    # Regression models
    "linreg": {"name": "Linear Regression", "type": "reg", "default_scaler": "standard"},
    "ridge": {"name": "Ridge Regression", "type": "reg", "default_scaler": "standard"},
    "lasso": {"name": "Lasso Regression", "type": "reg", "default_scaler": "standard"},
    "elastic": {"name": "Elastic Net", "type": "reg", "default_scaler": "standard"},
    "svr": {"name": "Support Vector Regression", "type": "reg", "default_scaler": "standard"},
    "knnr": {"name": "KNeighbors Regressor", "type": "reg", "default_scaler": "standard"},
    "rfr": {"name": "Random Forest Regressor", "type": "reg", "default_scaler": "none"},
    "xgbr": {"name": "XGBoost Regressor", "type": "reg", "default_scaler": "none"}
}

# Generate ALL_MODELS from the dictionary keys
ALL_MODELS = list(MODEL_DEFINITIONS.keys())
#options for the scaler dropdown menu
SCALER_OPTIONS = [
    {"label": "None","value": "none"},
    {"label": "StandardScaler","value": "standard"},
    {"label": "MinMaxScaler","value": "minmax"},
    {"label": "RobustScaler","value": "robust"},
    {"label": "MaxAbsScaler","value": "maxabs"},
]

PARAM_DEFAULTS = {
    "logreg": {"max_iter" : 1000, "random_state":42},
    "rf": {  "n_estimators":100, "max_depth":None, "random_state":42},
    "svc":{"probability": True, "random_state": 42, "gamma": 'scale'},
    "knnc":{"n_neighbors":5},
    "nb":{"var_smoothing": 1e-9},
    "adac":{"n_estimators":50, "random_state":42},
    "xgbc":{"eval_metric":'logloss', "random_state":42},
    #Regression Models
    "linreg": {"fit_intercept":True, "copy_X":True, "n_jobs":-1},
    "lasso": {"alpha": 1.0, "max_iter": 1000, "random_state": 42},
    "ridge": {"alpha": 1.0, "random_state": 42},
    "elastic": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000, "random_state": 42},
    "rfr": {"n_estimators": 100, "max_depth": None, "random_state": 42},
    "svr": {"kernel": 'rbf', "C": 1.0, "gamma": 'scale', "epsilon": 0.1},
    "knnr": {"n_neighbors": 5, "weights": 'uniform', "algorithm": 'auto'},
    "xgbr": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
}

PARAM_CHOICES = {
    "svc":  {"gamma": ["scale", "auto"]},
    "svr":  {"kernel": ["linear", "rbf", "poly", "sigmoid"], "gamma": ["scale", "auto"]},
    "knnr": {"weights": ["uniform", "distance"], "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]},
    "xgbc": {"eval_metric": ["logloss", "error", "auc"]},
    "rf":   {"max_depth": [None, 5, 10, 20]},
    "rfr":  {"max_depth": [None, 5, 10, 20]},
}

app.layout = html.Div(
    style={"fontFamily": "Arial, sans-serif", "margin": "20px"},
    children=[
        html.H3("Machine Learning Pipeline Dashboard"),

        # File upload menu
        dcc.Upload(
            id="upload-data",
            children=html.Button("Upload CSV"),
            style={
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "padding": "10px",
                "textAlign": "center",
                "width": "200px",
            },
            multiple=False,
        ),
        html.Div(id="file-name", style={"marginTop": "8px", "color": "green"}),

        html.Br(),
         # Target‐column input menu (allows user to choose target column otherwise chooses last column in dataset)
        html.Label("Target column name (leave blank -> use last column):"),
        dcc.Input(
            id="target-col-input",
            type="text",
            placeholder="e.g. 'price' or 'label'",
            style={"width": "300px", "marginBottom": "20px"},
        ),

        html.Br(),
        #Drop down menu for test/training split
        html.Label("Train/Test Split Ratio:"),
        create_help_icon("split-ratio-dropdown"),
        dcc.Dropdown(
            id="split-ratio-dropdown",
            options=[
                {"label": "70 / 30", "value": 0.7},
                {"label": "75 / 25", "value": 0.75},
                {"label": "80 / 20", "value": 0.8},
                {"label": "90 / 10", "value": 0.9},
            ],
            value=0.8,  # default to an 80/20 split
            clearable=False,
            style={"width": "150px", "marginBottom": "20px"},
        ),
        
        

        html.Br(),
        # Checkbox option to include a validation set
        html.Label("Validation Set:"),
        create_help_icon("include-val-checklist"),
        dcc.Checklist(
            id="include-val-checklist",
            options=[{"label": "Include validation set (20% of training)", "value": "yes"}],
            value=[],  # default: no
            labelStyle={"display": "inline-block"},
            style={"marginBottom": "20px"},
        ),

        html.Br(),
        #Menu to choose whether classification or regression task
        html.Label("Task Type:"),
        dcc.RadioItems(
            id="task-type",
            options=[
                {"label": "Classification", "value": "clf"},
                {"label": "Regression",     "value": "reg"},
            ],
            value="clf",  # default
            labelStyle={"display": "inline-block", "marginRight": "20px"},
            style={"marginBottom": "20px"},
        ),
        # Model selection
        html.Label("Choose Models:"),
        dcc.Checklist(
            id="model-choice",
            options=[],
            value=[],
            labelStyle={"display": "block"},
            style={"marginBottom": "20px"},
        ),
         # ---  Per‐model scaler dropdowns (hidden unless the corresponding model is checked) ---
         # Model scalers - generated from dictionary
         create_scaler_section_header(),
         *[create_Scaler_Dropdown(model_id, model_info["name"], model_info["default_scaler"]) for model_id, model_info in MODEL_DEFINITIONS.items()],
        #Param tag
        html.Div(id="param-panels", style={"marginTop": "10px"}),
        
        html.Br(),

        # Run button
        html.Button("Run", id="run-button", n_clicks=0),
        
        # Loading message (initially hidden)
        html.Div(
            id="loading-message",
            children="Training models, please wait...",
            style={"display": "none", "color": "blue", "fontSize": "16px", "margin": "10px 0"}
        ),

        html.Hr(),

        # Output area (metrics, messages, graphs)
        html.Div(id="output-div", style={"marginTop": "20px"}),
    ],
)

 

def parse_csv(contents):
    #Parses the uploaded CSV's contents into a Dataframe
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    # Assume CSV for simplicity
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    return df

# call back for regression vs classification
@app.callback(
    Output("model-choice", "options"),
    Input("task-type", "value"),
)
def set_model_options(task):
    # Filter models based on type directly from dictionary
    return [{"label": model_info["name"], "value": model_id} 
            for model_id, model_info in MODEL_DEFINITIONS.items()
            if model_info["type"] == task]

# callback for showing training models message
@app.callback(
    Output("loading-message", "style"),
    Input("run-button", "n_clicks"),
    prevent_initial_call=True
)
def show_loading_message(n_clicks):
    if n_clicks > 0:
        return {"display": "block", "color": "blue", "fontSize": "16px", "margin": "10px 0"}
    return {"display": "none"}
# ----------------------------------------------------------------------
# Callback to show/hide each model’s scaler dropdown
# ----------------------------------------------------------------------
@app.callback(
    [Output(f"scaler-{model}-container", "style") for model in ALL_MODELS],
    Input("model-choice", "value"),
    Input("task-type", "value"),
)
def toggle_scaler_containers(selected_models,task_type):
    #Show/hide scaler dropdowns based on selected models
    selected_models = [i for i in selected_models if MODEL_DEFINITIONS[i]["type"] == task_type] #makes sure that scaler dropdown matches current task type
    styles = []
    for model in ALL_MODELS:
        if model in selected_models:
            styles.append({"display": "block", "marginBottom": "20px"})
        else:
            styles.append({"display": "none"})
    return styles

@app.callback(
    Output("param-panels", "children"),
    Input("model-choice", "value"),
    Input("task-type", "value"),
)
def render_param_panels(selected_models, task_type):
    if not selected_models:
        return []
    # Only show param UIs for models of the current task type
    filtered = [m for m in selected_models if MODEL_DEFINITIONS[m]["type"] == task_type]
    return [build_param_section(m) for m in filtered]

@app.callback(
    Output("file-name", "children"),
    Input("upload-data", "filename"),
)
def display_filename(filename):
    if filename is None:
        return ""
    return f"Uploaded: {filename}"


@app.callback(
    Output("output-div", "children"),
    Input("run-button", "n_clicks"),
    State("upload-data", "contents"),
    State("target-col-input", "value"),
    State("split-ratio-dropdown", "value"),
    State("include-val-checklist", "value"),
    State("model-choice", "value"),
    State("task-type","value"),
    State({"type": "param", "model": ALL, "name": ALL}, "value"),
    State({"type": "param", "model": ALL, "name": ALL}, "id"),
    *[State(f"scaler-{model}", "value") for model in ALL_MODELS],
)

def run_models(
    n_clicks,
    contents,
    target_col,
    split_ratio,
    include_val_list,
    selected_models,
    task_type,
    param_values, 
    param_ids,
    *scaler_values,
):
    
    if n_clicks < 1:
        return ""
    if contents is None:
        return html.Div(" Please upload a CSV file first.", style={"color": "red"})


    # Load DataFrame from upload and check if it is clean
    try:
        df = parse_csv(contents)
        is_valid, error_msg = validate_clean_dataset(df,task_type)
        if not is_valid:
            return html.Div(error_msg, style={"color": "red", "fontSize": "16px", "padding": "10px"})
    except Exception as e:
        return html.Div(f"Error reading CSV: {e}", style={"color": "red"})

    if "yes" in include_val_list: #check if User wants to include a validation set
        include_val = True
    else:
        include_val = False
    #removes possibility that user would try and train regression and classification models at same time   
    filtered_models = [i for i in selected_models if MODEL_DEFINITIONS[i]["type"] == task_type]
    
    # Create a dictionary mapping model names to their scaler values
    scaler_dict = dict(zip(ALL_MODELS, scaler_values))
    
    #create dictionary mapping out hyperparams
    params_by_model = {}
    for val, pid in zip(param_values or [], param_ids or []):
        model = pid["model"]; name = pid["name"]
        default = PARAM_DEFAULTS[model][name]             # use real defaults for typing
        val = _decode_none(default, val)                  # map sentinel back to None (if applicable)
        coerced = _coerce_bool(default, val)              # normalize booleans: Dash true/false is True and [] so converts this into boolean
        #if None or empty string is found resort back to default values
        if coerced in (None, "", []):
            coerced = default
        if isinstance(default, int) and isinstance(coerced, float):
            coerced = int(coerced)
        params_by_model.setdefault(model, {})[name] = coerced

    # ---------- merge with defaults so each model has a full param dict ----------
    for m in selected_models or []:
        defaults = PARAM_DEFAULTS[m]
        edited = params_by_model.get(m, {})
        params_by_model[m] = {k: edited.get(k, v) for k, v in defaults.items()}
    print(params_by_model)   
    
    # Call custom function to split data, returns dictionary of split data
    split_dict = split_data(df, target_col, split_ratio, include_val)

    # unpack split data
    X_train = split_dict["X_train"]
    X_val = split_dict["X_val"]
    X_test = split_dict["X_test"]
    y_train = split_dict["y_train"]
    y_val = split_dict["y_val"]
    y_test = split_dict["y_test"]

    #classes is list of unique values in y_test
    classes = np.unique(y_test)

    #Prepare basic info to display
    info_lines = []
    fill_info_lines(df,target_col,include_val,X_train,X_test,X_val,info_lines)
    
    # For each selected model run metrics
    results = []
    for m in filtered_models:
        # Get the chosen scaler or use default from MODEL_DEFINITIONS
        if scaler_dict[m] is not None:
            chosen_scaler = scaler_dict[m]  # Could be "none", "standard", etc.
        else:
            chosen_scaler = MODEL_DEFINITIONS[m]["default_scaler"]
        
        # Get the model display name
        model_name = MODEL_DEFINITIONS[m]["name"]
        
        # Check if classification or regression
        if MODEL_DEFINITIONS[m]["type"] == "clf":
            model_out = trainClassification(X_train, X_test, X_val, y_train, y_test, y_val, chosen_scaler, m ,params_by_model)
            
            # Handle XGBoost encoding issue
            if m == "xgbc" and model_out["label_encoder"] is not None:
                y_test_for_output = model_out["label_encoder"].transform(y_test)
            else:
                y_test_for_output = y_test
                
            output_collector(model_name, model_out, y_test_for_output, X_train, classes, results)
            
        else:  # Regression
            model_out = trainRegression(X_train, X_test, X_val, y_train, y_test, y_val, chosen_scaler, m,params_by_model)
            output_collector_regression(model_name, model_out, y_test, X_train, results)
    

    if not results:
        results = [html.Li("No models selected.")]

    # Assembles everything here
    return html.Div(
        [
            #html.Div(id="loading-message", style={"display": "none"}),
            html.H3("Data Info"),
            *info_lines,
            html.Br(),
            html.H3("Model Results"),
            html.Hr(),
            html.Ul(results),
            # 5) Later, you can append real metrics or dcc.Graph(...) components here
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8051)
    print("use: http://localhost:8051/")

print("blah")



