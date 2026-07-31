import os
import io
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import gpflow
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

app = FastAPI(title="GP Training App Web")

# Global session state storing model and current data
SESSION_STATE: Dict[str, Any] = {
    "type_data": "Manual",
    "train_done": False,
    "model": None,
    "parms_X": None,
    "parms_Y": None,
    "X_Train": None,
    "Y_Train": None,
    "X_Test": None,
    "Y_Test": None,
    "X": None,
    "Y": None,
    "ncol_data": 2,
    "Axes_titles": ["X", "Y"],
    "Graph_title": "Manual Points Graph",
    "var_norm_label": "None",
    "var_norm_feat": "None",
    "var_kernel": "RBF",
    "trainlikelihood": False,
    "white_kernel": False,
    "Train_Test_Split": False,
    "Split_Percentage": 20.0,
    "num_params": 0,
    "df_full": None,
    "available_data_search": None
}

# =============================================================================
# NORMALIZATION FUNCTION (Exact match to original Tkinter script)
# =============================================================================
def Normalization(inpt, option, parms=None, reverse=False, var={"bol": False, "Y_N": None}):
    if inpt is None or len(inpt) == 0:
        return inpt, parms

    inpt = np.array(inpt, dtype=np.float64)

    if not var["bol"]:
        if option == "None":
            if not reverse:
                if parms is None:
                    mean = inpt.mean(axis=0)
                    std = inpt.std(axis=0)
                    parms = [mean, std]
                outpt = inpt
            else:
                outpt = inpt

        elif option == "Standardization":
            if not reverse:
                if parms is None:
                    mean = inpt.mean(axis=0)
                    std = inpt.std(axis=0)
                    # avoid zero division
                    std = np.where(std == 0, 1.0, std)
                    parms = [mean, std]
                outpt = (inpt - parms[0]) / parms[1]
            else:
                outpt = (inpt * parms[1]) + parms[0]

        elif option == "LogStand":
            if not reverse:
                if parms is None:
                    mean = np.log(inpt).mean(axis=0)
                    std = np.log(inpt).std(axis=0)
                    std = np.where(std == 0, 1.0, std)
                    parms = [mean, std]
                outpt = (np.log(inpt) - parms[0]) / parms[1]
            else:
                outpt = np.exp(inpt * parms[1] + parms[0])

        elif option == "Log + bStand":
            b = 10**-3
            if not reverse:
                if parms is None:
                    mean = np.log(inpt + b).mean(axis=0)
                    std = np.log(inpt + b).std(axis=0)
                    std = np.where(std == 0, 1.0, std)
                    parms = [mean, std]
                outpt = (np.log(inpt + b) - parms[0]) / parms[1]
            else:
                outpt = np.exp(inpt * parms[1] + parms[0]) - b

        elif option == "MinMax":
            if not reverse:
                if parms is None:
                    inpt_min = np.min(inpt, axis=0)
                    inpt_max = np.max(inpt, axis=0)
                    diff = inpt_max - inpt_min
                    diff = np.where(diff == 0, 1.0, diff)
                    parms = [inpt_min, inpt_max]
                outpt = (inpt - parms[0]) / (parms[1] - parms[0])
            else:
                outpt = inpt * (parms[1] - parms[0]) + parms[0]

    elif var["bol"]:
        if option == "None":
            if reverse:
                outpt = inpt

        elif option == "Standardization":
            if reverse:
                outpt = inpt * (parms[1] ** 2)

        elif option == "LogStand":
            if reverse:
                outpt = inpt * (parms[1] * np.exp(parms[0] + parms[1] * var["Y_N"])) ** 2

        elif option == "Log + bStand":
            if reverse:
                outpt = inpt * (parms[1] * np.exp(parms[0] + parms[1] * var["Y_N"])) ** 2

        elif option == "MinMax":
            if reverse:
                outpt = inpt * ((parms[1] - parms[0]) ** 2)

    return outpt, parms

# =============================================================================
# API MODELS
# =============================================================================
class ManualTrainData(BaseModel):
    x_points: List[float]
    y_points: List[float]
    kernel: str = "RBF"
    norm_label: str = "None"
    norm_feat: str = "None"
    likelihood: bool = False
    white_kernel: bool = False
    split: bool = False
    split_percentage: float = 20.0

class CSVTrainData(BaseModel):
    label_col: str
    feature_cols: List[str]
    kernel: str = "RBF"
    norm_label: str = "None"
    norm_feat: str = "None"
    likelihood: bool = False
    white_kernel: bool = False
    split: bool = False
    split_percentage: float = 20.0

class PredictYRequest(BaseModel):
    x_values: List[float]
    confidence_level: float = 95.0

class PlotGraphRequest(BaseModel):
    standard_plot: bool = True
    n_points: int = 1000
    var_ranges: Optional[List[Dict[str, float]]] = None  # [{'min': 0, 'max': 10}, ...]

class ALBORequest(BaseModel):
    af_type: str = "Std"  # "Std", "Std/Mean", "PI", "EI", "UCB"
    standard_plot: bool = True
    import_available: bool = False
    n_points: int = 1000
    x_ranges: Optional[List[Dict[str, float]]] = None

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8-sig", errors="ignore")
    lines = text.splitlines()

    comment_lines_idx = []
    comment_lines_text = []
    for i, line in enumerate(lines):
        if line.strip().startswith("#"):
            comment_lines_idx.append(i)
            comment_lines_text.append(line.strip())
        else:
            break

    graph_title = "Uploaded CSV Data"
    if comment_lines_text:
        graph_title = comment_lines_text[0][1:].strip(",")

    df = pd.read_csv(io.StringIO(text), skiprows=comment_lines_idx)
    columns = df.columns.tolist()

    SESSION_STATE["df_full"] = df
    SESSION_STATE["Graph_title"] = graph_title

    return {
        "columns": columns,
        "comments": comment_lines_text,
        "graph_title": graph_title,
        "preview": df.head(10).to_dict(orient="records")
    }

def fit_gp_model(X, Y, config):
    var_kernel = config["kernel"]
    var_norm_label = config["norm_label"]
    var_norm_feat = config["norm_feat"]
    trainlikelihood = config["likelihood"]
    white_kernel = config["white_kernel"]
    split = config["split"]
    split_percentage = config["split_percentage"]

    if split and len(X) > 2:
        X_Train, X_Test, Y_Train, Y_Test = train_test_split(
            X, Y, random_state=11, test_size=split_percentage / 100.0
        )
        Train_Test_Split = True
    else:
        X_Train = X
        Y_Train = Y
        X_Test = np.array([]).reshape(0, X.shape[1])
        Y_Test = np.array([]).reshape(0, 1)
        Train_Test_Split = False

    # Normalization
    X_Train_N, parms_X = Normalization(X_Train, var_norm_feat, parms=None, reverse=False)
    Y_Train_N, parms_Y = Normalization(Y_Train, var_norm_label, parms=None, reverse=False)

    # Kernel setup
    if var_kernel == "RBF":
        kernel_in_use = gpflow.kernels.SquaredExponential()
    elif var_kernel == "Matern 12":
        kernel_in_use = gpflow.kernels.Matern12()
    elif var_kernel == "Matern 32":
        kernel_in_use = gpflow.kernels.Matern32()
    elif var_kernel == "Matern 52":
        kernel_in_use = gpflow.kernels.Matern52()
    elif var_kernel == "Periodic":
        kernel_in_use = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential())
    elif var_kernel == "RQ":
        kernel_in_use = gpflow.kernels.RationalQuadratic()
    else:
        kernel_in_use = gpflow.kernels.SquaredExponential()

    if white_kernel:
        kernel_in_use = kernel_in_use + gpflow.kernels.White()

    # Fit Model
    model = gpflow.models.GPR((X_Train_N, Y_Train_N), kernel=kernel_in_use, noise_variance=10**-5)
    gpflow.utilities.set_trainable(model.likelihood.variance, trainlikelihood)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables)

    num_params = sum(p.shape.num_elements() for p in model.trainable_parameters)

    # Save to session
    SESSION_STATE["model"] = model
    SESSION_STATE["parms_X"] = parms_X
    SESSION_STATE["parms_Y"] = parms_Y
    SESSION_STATE["train_done"] = True
    SESSION_STATE["X_Train"] = X_Train
    SESSION_STATE["Y_Train"] = Y_Train
    SESSION_STATE["X_Test"] = X_Test
    SESSION_STATE["Y_Test"] = Y_Test
    SESSION_STATE["X"] = X
    SESSION_STATE["Y"] = Y
    SESSION_STATE["ncol_data"] = X.shape[1] + 1
    SESSION_STATE["var_norm_label"] = var_norm_label
    SESSION_STATE["var_norm_feat"] = var_norm_feat
    SESSION_STATE["var_kernel"] = var_kernel
    SESSION_STATE["trainlikelihood"] = trainlikelihood
    SESSION_STATE["white_kernel"] = white_kernel
    SESSION_STATE["Train_Test_Split"] = Train_Test_Split
    SESSION_STATE["Split_Percentage"] = split_percentage
    SESSION_STATE["num_params"] = num_params

    return model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params

@app.post("/api/train-manual")
async def train_manual(req: ManualTrainData):
    x_valid = [x for x, y in zip(req.x_points, req.y_points) if x is not None and y is not None]
    y_valid = [y for x, y in zip(req.x_points, req.y_points) if x is not None and y is not None]

    if len(x_valid) == 0:
        raise HTTPException(status_code=400, detail="Please enter at least 1 pair of training points.")

    X = np.array(x_valid, dtype=np.float64).reshape(-1, 1)
    Y = np.array(y_valid, dtype=np.float64).reshape(-1, 1)

    SESSION_STATE["type_data"] = "Manual"
    SESSION_STATE["Axes_titles"] = ["X", "Y"]
    SESSION_STATE["Graph_title"] = "Manual Points Graph (for testing purposes)"

    config = {
        "kernel": req.kernel,
        "norm_label": req.norm_label,
        "norm_feat": req.norm_feat,
        "likelihood": req.likelihood,
        "white_kernel": req.white_kernel,
        "split": req.split,
        "split_percentage": req.split_percentage
    }

    model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params = fit_gp_model(X, Y, config)

    return {
        "status": "success",
        "n_dimensions": 2,
        "n_train": int(len(X_Train)),
        "n_test": int(len(X_Test)),
        "num_params": int(num_params),
        "axes_titles": ["X", "Y"]
    }

@app.post("/api/train-csv")
async def train_csv(req: CSVTrainData):
    df = SESSION_STATE.get("df_full")
    if df is None:
        raise HTTPException(status_code=400, detail="No CSV file uploaded yet.")

    if req.label_col not in df.columns:
        raise HTTPException(status_code=400, detail="Selected label column not found in CSV.")

    selected_cols = [c for c in req.feature_cols if c in df.columns and c != req.label_col]
    if len(selected_cols) == 0:
        raise HTTPException(status_code=400, detail="Select at least 1 feature column.")

    df_feat = df[selected_cols]
    df_label = df[req.label_col]
    df_new = pd.concat([df_feat, df_label], axis=1).dropna()

    Col_names = list(df_new.columns)
    Axes_titles = [n[:10] + "..." if len(n) > 10 else n for n in Col_names]

    ncol_data = df_new.shape[1]
    X = df_new.iloc[:, :-1].values.astype(float)
    Y = df_new.iloc[:, -1].values.reshape(-1, 1).astype(float)

    SESSION_STATE["type_data"] = "Import"
    SESSION_STATE["Axes_titles"] = Axes_titles

    config = {
        "kernel": req.kernel,
        "norm_label": req.norm_label,
        "norm_feat": req.norm_feat,
        "likelihood": req.likelihood,
        "white_kernel": req.white_kernel,
        "split": req.split,
        "split_percentage": req.split_percentage
    }

    model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params = fit_gp_model(X, Y, config)

    return {
        "status": "success",
        "n_dimensions": int(ncol_data),
        "n_train": int(len(X_Train)),
        "n_test": int(len(X_Test)),
        "num_params": int(num_params),
        "axes_titles": Axes_titles
    }

@app.get("/api/parity")
async def get_parity_data():
    if not SESSION_STATE["train_done"]:
        return {"train_done": False}

    model = SESSION_STATE["model"]
    parms_X = SESSION_STATE["parms_X"]
    parms_Y = SESSION_STATE["parms_Y"]
    X_Train = SESSION_STATE["X_Train"]
    Y_Train = SESSION_STATE["Y_Train"]
    X_Test = SESSION_STATE["X_Test"]
    Y_Test = SESSION_STATE["Y_Test"]
    Y = SESSION_STATE["Y"]
    var_norm_label = SESSION_STATE["var_norm_label"]
    var_norm_feat = SESSION_STATE["var_norm_feat"]
    Axes_titles = SESSION_STATE["Axes_titles"]
    has_test = SESSION_STATE["Train_Test_Split"] and len(X_Test) > 0

    X_Train_N, _ = Normalization(X_Train, var_norm_feat, parms=parms_X, reverse=False)
    Y_Train_pred_N, Y_Train_pred_std_N = model.predict_y(X_Train_N, full_cov=False)

    Y_Train_pred = Normalization(np.array(Y_Train_pred_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
    Y_Train_pred_std = Normalization(np.array(Y_Train_pred_std_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_Train_pred_N})[0]

    metrics_dict = {
        "R2_Train": float(metrics.r2_score(Y_Train, Y_Train_pred)),
        "RMSE_Train": float(metrics.root_mean_squared_error(Y_Train, Y_Train_pred)),
        "MAE_Train": float(metrics.mean_absolute_error(Y_Train, Y_Train_pred)),
        "MAPE_Train": float(metrics.mean_absolute_percentage_error(Y_Train, Y_Train_pred))
    }

    test_data = None
    if has_test:
        X_Test_N, _ = Normalization(X_Test, var_norm_feat, parms=parms_X, reverse=False)
        Y_Test_pred_N, Y_Test_pred_std_N = model.predict_y(X_Test_N, full_cov=False)
        Y_Test_pred = Normalization(np.array(Y_Test_pred_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
        Y_Test_pred_std = Normalization(np.array(Y_Test_pred_std_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_Test_pred_N})[0]

        metrics_dict.update({
            "R2_Test": float(metrics.r2_score(Y_Test, Y_Test_pred)),
            "RMSE_Test": float(metrics.root_mean_squared_error(Y_Test, Y_Test_pred)),
            "MAE_Test": float(metrics.mean_absolute_error(Y_Test, Y_Test_pred)),
            "MAPE_Test": float(metrics.mean_absolute_percentage_error(Y_Test, Y_Test_pred))
        })

        test_data = {
            "y_exp": Y_Test.flatten().tolist(),
            "y_pred": Y_Test_pred.flatten().tolist(),
            "y_std": np.sqrt(np.maximum(0, Y_Test_pred_std.flatten())).tolist()
        }

    y_min = float(min(Y.flatten()))
    y_max = float(max(Y.flatten()))

    return {
        "train_done": True,
        "has_test": has_test,
        "axes_titles": Axes_titles,
        "train": {
            "y_exp": Y_Train.flatten().tolist(),
            "y_pred": Y_Train_pred.flatten().tolist(),
            "y_std": np.sqrt(np.maximum(0, Y_Train_pred_std.flatten())).tolist()
        },
        "test": test_data,
        "line_min": y_min,
        "line_max": y_max,
        "metrics": metrics_dict
    }

@app.post("/api/plot-graph")
async def get_plot_graph(req: PlotGraphRequest):
    if not SESSION_STATE["train_done"]:
        return {"train_done": False}

    model = SESSION_STATE["model"]
    parms_X = SESSION_STATE["parms_X"]
    parms_Y = SESSION_STATE["parms_Y"]
    X_Train = SESSION_STATE["X_Train"]
    Y_Train = SESSION_STATE["Y_Train"]
    X_Test = SESSION_STATE["X_Test"]
    Y_Test = SESSION_STATE["Y_Test"]
    X = SESSION_STATE["X"]
    ncol_data = SESSION_STATE["ncol_data"]
    var_norm_label = SESSION_STATE["var_norm_label"]
    var_norm_feat = SESSION_STATE["var_norm_feat"]
    Axes_titles = SESSION_STATE["Axes_titles"]
    Graph_title = SESSION_STATE["Graph_title"]
    has_test = SESSION_STATE["Train_Test_Split"] and len(X_Test) > 0

    n_features = ncol_data - 1
    N_Points = round(int(req.n_points) ** (1 / max(1, n_features)))

    X_Plot_base = np.zeros((N_Points, n_features))

    if req.standard_plot:
        for n in range(n_features):
            varRange = np.linspace(X[:, n].min(), X[:, n].max(), N_Points)
            X_Plot_base[:, n] = varRange.copy()
    else:
        if req.var_ranges and len(req.var_ranges) >= n_features:
            for n in range(n_features):
                v_min = req.var_ranges[n]["min"]
                v_max = req.var_ranges[n]["max"]
                varRange = np.linspace(v_min, v_max, N_Points)
                X_Plot_base[:, n] = varRange.copy()
        else:
            for n in range(n_features):
                varRange = np.linspace(X[:, n].min(), X[:, n].max(), N_Points)
                X_Plot_base[:, n] = varRange.copy()

    if n_features == 1:
        X_Plot = X_Plot_base
    else:
        X_Plot = np.array(np.meshgrid(*[X_Plot_base[:, i] for i in range(n_features)])).T.reshape(-1, n_features)

    X_Plot_N, _ = Normalization(X_Plot, var_norm_feat, parms=parms_X, reverse=False)
    Y_mean_N, Y_var_N = model.predict_y(X_Plot_N, full_cov=False)

    Y_mean = Normalization(np.array(Y_mean_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
    Y_var = Normalization(np.array(Y_var_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_mean_N})[0]

    result = {
        "train_done": True,
        "n_features": n_features,
        "axes_titles": Axes_titles,
        "graph_title": Graph_title,
        "has_test": has_test,
        "train_points": {
            "x": X_Train.tolist(),
            "y": Y_Train.flatten().tolist()
        },
        "test_points": {
            "x": X_Test.tolist(),
            "y": Y_Test.flatten().tolist()
        } if has_test else None
    }

    if n_features == 1:
        Y_Upper = Y_mean.flatten() + 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))
        Y_Lower = Y_mean.flatten() - 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))
        result.update({
            "x_plot": X_Plot.flatten().tolist(),
            "y_mean": Y_mean.flatten().tolist(),
            "y_upper": Y_Upper.tolist(),
            "y_lower": Y_Lower.tolist()
        })
    elif n_features == 2:
        result.update({
            "x1_plot": X_Plot[:, 0].tolist(),
            "x2_plot": X_Plot[:, 1].tolist(),
            "y_mean": Y_mean.flatten().tolist()
        })

    return result

@app.post("/api/predict-y")
async def predict_y(req: PredictYRequest):
    if not SESSION_STATE["train_done"]:
        raise HTTPException(status_code=400, detail="Model is not trained yet.")

    model = SESSION_STATE["model"]
    parms_X = SESSION_STATE["parms_X"]
    parms_Y = SESSION_STATE["parms_Y"]
    var_norm_label = SESSION_STATE["var_norm_label"]
    var_norm_feat = SESSION_STATE["var_norm_feat"]

    x_val = np.array(req.x_values, dtype=np.float64).reshape(1, -1)
    x_val_N, _ = Normalization(x_val, var_norm_feat, parms=parms_X, reverse=False)

    Y_mean_N, Y_var_N = model.predict_y(x_val_N, full_cov=False)

    Y_mean = Normalization(np.array(Y_mean_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
    Y_var = Normalization(np.array(Y_var_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_mean_N})[0]

    conf_level = req.confidence_level / 100.0
    z_score = norm.ppf(1 - (1 - conf_level) / 2.0)

    y_val = float(Y_mean[0, 0])
    std_val = float(np.sqrt(max(0, Y_var[0, 0])))
    y_lower = y_val - z_score * std_val
    y_upper = y_val + z_score * std_val

    return {
        "pred_y": round(y_val, 3),
        "std_y": round(std_val, 3),
        "ci_lower": round(y_lower, 3),
        "ci_upper": round(y_upper, 3),
        "confidence_level": req.confidence_level
    }

@app.post("/api/albo")
async def run_albo(req: ALBORequest):
    if not SESSION_STATE["train_done"]:
        raise HTTPException(status_code=400, detail="Model is not trained yet.")

    model = SESSION_STATE["model"]
    parms_X = SESSION_STATE["parms_X"]
    parms_Y = SESSION_STATE["parms_Y"]
    X = SESSION_STATE["X"]
    ncol_data = SESSION_STATE["ncol_data"]
    var_norm_label = SESSION_STATE["var_norm_label"]
    var_norm_feat = SESSION_STATE["var_norm_feat"]
    Axes_titles = SESSION_STATE["Axes_titles"]
    Graph_title = SESSION_STATE["Graph_title"]

    n_features = ncol_data - 1
    if n_features < 1 or n_features > 2:
        raise HTTPException(status_code=400, detail="AL/BO search plot supports 1D and 2D features.")

    if not req.import_available:
        if req.standard_plot:
            x_min = np.array([np.min(X[:, e]) for e in range(n_features)]).reshape(-1, 1)
            x_max = np.array([np.max(X[:, e]) for e in range(n_features)]).reshape(-1, 1)
        else:
            if req.x_ranges and len(req.x_ranges) >= n_features:
                x_min = np.array([req.x_ranges[e]["min"] for e in range(n_features)]).reshape(-1, 1)
                x_max = np.array([req.x_ranges[e]["max"] for e in range(n_features)]).reshape(-1, 1)
            else:
                x_min = np.array([np.min(X[:, e]) for e in range(n_features)]).reshape(-1, 1)
                x_max = np.array([np.max(X[:, e]) for e in range(n_features)]).reshape(-1, 1)

        N_Points = round(int(req.n_points) ** (1 / n_features))
        BO_zone_base = np.zeros((N_Points, n_features))

        for n in range(n_features):
            varRange = np.linspace(float(x_min[n]), float(x_max[n]), N_Points)
            BO_zone_base[:, n] = varRange.copy()

        if n_features == 1:
            BO_zone = BO_zone_base
        else:
            BO_zone = np.array(np.meshgrid(*[BO_zone_base[:, i] for i in range(n_features)])).T.reshape(-1, n_features)

        BO_zone_N, _ = Normalization(BO_zone, var_norm_feat, parms=parms_X, reverse=False)
    else:
        df = SESSION_STATE.get("df_full")
        if df is None:
            raise HTTPException(status_code=400, detail="No available dataset uploaded for search zone.")
        BO_zone = df.iloc[:, :n_features].values.astype(float)
        BO_zone_N, _ = Normalization(BO_zone, var_norm_feat, parms=parms_X, reverse=False)

    Y_mean_N, Y_var_N = model.predict_y(BO_zone_N, full_cov=False)

    Y_mean = Normalization(np.array(Y_mean_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
    Y_var = Normalization(np.array(Y_var_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_mean_N})[0]

    Y_mean_flat = Y_mean.flatten()
    Y_var_flat = np.maximum(1e-12, Y_var.flatten())
    y_max_observed = float(np.max(Y_mean_flat))

    var_AF = req.af_type
    if var_AF == "PI":
        AF = norm.cdf((Y_mean_flat - y_max_observed) / Y_var_flat, 0, 1)
    elif var_AF == "EI":
        diff = Y_mean_flat - y_max_observed
        AF = diff * norm.cdf(diff / Y_var_flat, 0, 1) + Y_var_flat * norm.pdf(diff / Y_var_flat, 0, 1)
    elif var_AF == "UCB":
        lamda = 1.0
        AF = Y_mean_flat + lamda * Y_var_flat
    elif var_AF == "Std":
        AF = np.sqrt(Y_var_flat)
    elif var_AF == "Std/Mean":
        mean_safe = np.where(Y_mean_flat == 0, 1e-12, Y_mean_flat)
        AF = np.sqrt(Y_var_flat) / mean_safe
    else:
        AF = np.sqrt(Y_var_flat)

    max_AF_val = float(np.max(AF))
    idx_max = int(np.argmax(AF))
    next_point = BO_zone[idx_max].tolist()

    result = {
        "af_type": var_AF,
        "max_af": round(max_AF_val, 4),
        "next_point": [round(p, 3) for p in next_point],
        "n_features": n_features,
        "axes_titles": Axes_titles,
        "graph_title": Graph_title
    }

    if n_features == 1:
        result.update({
            "x_plot": BO_zone.flatten().tolist(),
            "y_mean": Y_mean_flat.tolist(),
            "af_plot": AF.tolist()
        })
    elif n_features == 2:
        result.update({
            "x1_plot": BO_zone[:, 0].tolist(),
            "x2_plot": BO_zone[:, 1].tolist(),
            "af_plot": AF.tolist()
        })

    return result

@app.post("/api/global-reset")
async def global_reset():
    SESSION_STATE.update({
        "type_data": "Manual",
        "train_done": False,
        "model": None,
        "parms_X": None,
        "parms_Y": None,
        "X_Train": None,
        "Y_Train": None,
        "X_Test": None,
        "Y_Test": None,
        "X": None,
        "Y": None,
        "ncol_data": 2,
        "Axes_titles": ["X", "Y"],
        "num_params": 0
    })
    return {"status": "reset_success"}

# Serve Frontend static assets
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse(content="<h1>GP Training App Web</h1><p>Index file missing.</p>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
