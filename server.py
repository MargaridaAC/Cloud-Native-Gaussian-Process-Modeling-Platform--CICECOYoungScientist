import base64
import io
import json
import os
import tempfile
import time
import uuid
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gpflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.stats import norm
from sklearn import metrics
from sklearn.model_selection import train_test_split

app = FastAPI(title="GP Training App Web")

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none', dpi=130)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def get_safe_cmap(name: str | None) -> str:
    if not name:
        return "viridis"
    c_lower = str(name).strip().lower()
    try:
        plt.get_cmap(c_lower)
        return c_lower
    except Exception:
        return "viridis"


# =============================================================================
# HEALTH CHECK ENDPOINTS FOR ORCHESTRATION
# =============================================================================

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    return {"status": "ok"}

# =============================================================================
# MULTI-USER SESSION MANAGEMENT & STORAGE (DISK / REDIS)
# =============================================================================

SESSION_DIR = os.path.join(tempfile.gettempdir(), "gp_sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

# In-memory GPFlow model objects per session ID
SESSION_MODELS: dict[str, Any] = {}

REDIS_URL = os.environ.get("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        import redis
        redis_client = redis.from_url(REDIS_URL)
        print(f"Connected to Redis session store at {REDIS_URL}")
    except Exception as e:
        print(f"Failed to connect to Redis at {REDIS_URL}: {e}")

def get_default_session_state() -> dict[str, Any]:
    return {
        "type_data": "Manual",
        "train_done": False,
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
        "Graph_title": "Manual Points Graph (for testing purposes)",
        "var_norm_label": "None",
        "var_norm_feat": "None",
        "var_kernel": "RBF",
        "trainlikelihood": False,
        "white_kernel": False,
        "Train_Test_Split": False,
        "Split_Percentage": 20.0,
        "num_params": 0,
        "df_full": None,
        "df_available": None,
        "BO_zone_available": None,
        "Save_path": None,
        "last_accessed": time.time()
    }

class SessionManager:
    @staticmethod
    def get_session_id(request: Request) -> str:
        sid = request.headers.get("X-Session-ID") or request.cookies.get("session_id")
        if not sid or not sid.strip():
            sid = str(uuid.uuid4())
        return sid.strip()

    @staticmethod
    def get_session(session_id: str) -> dict[str, Any]:
        SessionManager.cleanup_expired_sessions()

        state = None
        if redis_client:
            try:
                raw_data = redis_client.get(f"session:{session_id}")
                if raw_data:
                    state = json.loads(raw_data.decode("utf-8"))
            except Exception:
                pass

        if state is None:
            filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        state = json.load(f)
                except Exception:
                    state = None

        if state is None:
            state = get_default_session_state()

        state["last_accessed"] = time.time()

        # Restore numpy arrays if saved as lists
        for key in ["X", "Y", "X_Train", "Y_Train", "X_Test", "Y_Test"]:
            val = state.get(key)
            if val is not None:
                if isinstance(val, list):
                    arr = np.array(val, dtype=np.float64) if len(val) > 0 else None
                elif isinstance(val, np.ndarray):
                    arr = val
                else:
                    arr = None
                if arr is not None and arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                state[key] = arr

        if state.get("BO_zone_available") is not None and isinstance(state["BO_zone_available"], list):
            state["BO_zone_available"] = np.array(state["BO_zone_available"], dtype=np.float64) if len(state["BO_zone_available"]) > 0 else None

        if state.get("df_full") is not None and isinstance(state["df_full"], list):
            state["df_full"] = pd.DataFrame(state["df_full"])

        if state.get("df_available") is not None and isinstance(state["df_available"], list):
            state["df_available"] = pd.DataFrame(state["df_available"])

        state["model"] = SESSION_MODELS.get(session_id)
        return state

    @staticmethod
    def save_session(session_id: str, state: dict[str, Any]):
        state["last_accessed"] = time.time()
        model = state.get("model")
        if model is not None:
            SESSION_MODELS[session_id] = model

        serializable_state = {}
        for k, v in state.items():
            if k == "model":
                continue
            elif isinstance(v, np.ndarray):
                serializable_state[k] = v.tolist()
            elif isinstance(v, pd.DataFrame):
                serializable_state[k] = v.to_dict(orient="records")
            else:
                serializable_state[k] = v

        json_str = json.dumps(serializable_state, indent=2)

        if redis_client:
            try:
                redis_client.setex(f"session:{session_id}", 3600, json_str)
            except Exception:
                pass

        filepath = os.path.join(SESSION_DIR, f"{session_id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        except Exception as e:
            print(f"Error saving session state: {e}")

    @staticmethod
    def cleanup_expired_sessions(max_age_seconds: int = 3600):
        now = time.time()
        try:
            for filename in os.listdir(SESSION_DIR):
                if filename.endswith(".json"):
                    filepath = os.path.join(SESSION_DIR, filename)
                    try:
                        file_mtime = os.path.getmtime(filepath)
                        if now - file_mtime > max_age_seconds:
                            os.remove(filepath)
                            sid = filename[:-5]
                            SESSION_MODELS.pop(sid, None)
                    except Exception:
                        pass
        except Exception:
            pass

# =============================================================================
# NORMALIZATION FUNCTION (Exact match to original script)
# =============================================================================

def Normalization(inpt, option, parms=None, reverse=False, var={"bol": False, "Y_N": None}):
    if inpt is None or len(inpt) == 0:
        return inpt, parms

    inpt = np.array(inpt, dtype=np.float64)
    if parms is not None:
        parms = [np.asarray(parms[0], dtype=np.float64), np.asarray(parms[1], dtype=np.float64)]

    outpt = inpt

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
            outpt = inpt

        elif option == "Standardization":
            if reverse:
                outpt = inpt * (parms[1] ** 2)

        elif option == "LogStand" or option == "Log + bStand":
            if reverse:
                outpt = inpt * (parms[1] * np.exp(parms[0] + parms[1] * var["Y_N"])) ** 2

        elif option == "MinMax":
            if reverse:
                outpt = inpt * ((parms[1] - parms[0]) ** 2)

    return outpt, parms

# =============================================================================
# API MODELS & SCHEMAS
# =============================================================================

class ManualTrainData(BaseModel):
    x_points: list[float]
    y_points: list[float]
    kernel: str = "RBF"
    norm_label: str = "None"
    norm_feat: str = "None"
    likelihood: bool = False
    white_kernel: bool = False
    split: bool = False
    split_percentage: float = 20.0

class CSVTrainData(BaseModel):
    label_col: str
    feature_cols: list[str]
    kernel: str = "RBF"
    norm_label: str = "None"
    norm_feat: str = "None"
    likelihood: bool = False
    white_kernel: bool = False
    split: bool = False
    split_percentage: float = 20.0

class AvailableDataConfirm(BaseModel):
    feature_cols: list[str]

class PredictYRequest(BaseModel):
    x_values: list[float]
    confidence_level: float = 95.0

class PlotGraphRequest(BaseModel):
    standard_plot: bool = True
    n_points: int = 1000
    var_ranges: list[dict[str, float]] | None = None
    cmap: str = "Viridis"
    model_color: str = "black"
    ic_color: str = "blue"

class ALBORequest(BaseModel):
    af_type: str = "Std"
    standard_plot: bool = True
    import_available: bool = False
    n_points: int = 1000
    x_ranges: list[dict[str, float]] | None = None
    cmap: str = "Viridis"
    model_color: str = "black"
    af_color: str = "blue"

class SessionSaveSchema(BaseModel):
    type_data: str | None = "Manual"
    train_done: bool | None = False
    ncol_data: int | None = 2
    Axes_titles: list[str] | None = ["X", "Y"]
    Graph_title: str | None = ""
    var_norm_label: str | None = "None"
    var_norm_feat: str | None = "None"
    var_kernel: str | None = "RBF"
    trainlikelihood: bool | None = False
    white_kernel: bool | None = False
    Train_Test_Split: bool | None = False
    Split_Percentage: float | None = 20.0
    num_params: int | None = 0
    X: list[Any] | None = None
    Y: list[Any] | None = None
    X_Train: list[Any] | None = None
    Y_Train: list[Any] | None = None
    X_Test: list[Any] | None = None
    Y_Test: list[Any] | None = None
    parms_X: list[Any] | None = None
    parms_Y: list[Any] | None = None
    df_full: list[dict[str, Any]] | None = None
    df_available: list[dict[str, Any]] | None = None
    BO_zone_available: list[Any] | None = None
    Save_path: str | None = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_serializable(val):
    if isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (list, tuple)):
        return [to_serializable(item) for item in val]
    elif isinstance(val, (np.generic, np.number)):
        return val.item()
    return val

def get_session_n_features(session_state: dict[str, Any]) -> int:
    X_Train = session_state.get("X_Train")
    if X_Train is not None:
        arr = np.asarray(X_Train)
        if arr.size > 0:
            return arr.shape[1] if arr.ndim > 1 else 1

    X = session_state.get("X")
    if X is not None:
        arr = np.asarray(X)
        if arr.size > 0:
            return arr.shape[1] if arr.ndim > 1 else 1

    ncol_data = session_state.get("ncol_data") or session_state.get("n_dimensions") or 2
    return max(1, int(ncol_data) - 1)

def fit_gp_model(X, Y, config, session_state: dict[str, Any]):
    session_state.pop("BO_zone_available", None)
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

    # Update session state
    session_state["model"] = model
    session_state["parms_X"] = to_serializable(parms_X)
    session_state["parms_Y"] = to_serializable(parms_Y)
    session_state["train_done"] = True
    session_state["X_Train"] = X_Train
    session_state["Y_Train"] = Y_Train
    session_state["X_Test"] = X_Test
    session_state["Y_Test"] = Y_Test
    session_state["X"] = X
    session_state["Y"] = Y
    session_state["ncol_data"] = X.shape[1] + 1
    session_state["var_norm_label"] = var_norm_label
    session_state["var_norm_feat"] = var_norm_feat
    session_state["var_kernel"] = var_kernel
    session_state["trainlikelihood"] = trainlikelihood
    session_state["white_kernel"] = white_kernel
    session_state["Train_Test_Split"] = Train_Test_Split
    session_state["Split_Percentage"] = split_percentage
    session_state["num_params"] = num_params

    return model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params

def get_var_bounds(session_state: dict[str, Any]):
    X = session_state.get("X")
    Axes_titles = session_state.get("Axes_titles", ["X", "Y"])
    if X is None or len(X) == 0:
        return [{"name": "X", "min": 0.0, "max": 10.0}]

    n_features = X.shape[1]
    bounds = []
    for i in range(n_features):
        name = Axes_titles[i] if i < len(Axes_titles) else f"Variable {i+1}"
        v_min = float(np.min(X[:, i]))
        v_max = float(np.max(X[:, i]))
        bounds.append({"name": name, "min": round(v_min, 4), "max": round(v_max, 4)})
    return bounds

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/model-info")
async def get_model_info(request: Request):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    n_features = get_session_n_features(session_state)
    X_Train = session_state.get("X_Train")
    X_Test = session_state.get("X_Test")
    n_train = len(X_Train) if X_Train is not None else 0
    n_test = len(X_Test) if X_Test is not None else 0

    return {
        "train_done": session_state.get("train_done", False),
        "type_data": session_state.get("type_data", "Manual"),
        "ncol_data": n_features + 1,
        "n_features": n_features,
        "axes_titles": session_state.get("Axes_titles", ["X", "Y"]),
        "graph_title": session_state.get("Graph_title", ""),
        "var_bounds": get_var_bounds(session_state),
        "num_params": session_state.get("num_params", 0),
        "n_train": n_train,
        "n_test": n_test,
        "save_path": session_state.get("Save_path")
    }

@app.post("/api/upload-csv")
async def upload_csv(request: Request, file: UploadFile = File(...)):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

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

    session_state["df_full"] = df
    session_state["Graph_title"] = graph_title
    SessionManager.save_session(sid, session_state)

    return {
        "columns": columns,
        "comments": comment_lines_text,
        "graph_title": graph_title,
        "preview": df.head(10).to_dict(orient="records")
    }

@app.post("/api/upload-available-data")
async def upload_available_data(request: Request, file: UploadFile = File(...)):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    contents = await file.read()
    text = contents.decode("utf-8-sig", errors="ignore")
    lines = text.splitlines()

    comment_lines_idx = [i for i, line in enumerate(lines) if line.strip().startswith("#")]
    df = pd.read_csv(io.StringIO(text), skiprows=comment_lines_idx)
    columns = df.columns.tolist()

    session_state["df_available"] = df
    SessionManager.save_session(sid, session_state)

    return {
        "columns": columns,
        "preview": df.head(10).to_dict(orient="records")
    }

@app.post("/api/confirm-available-data")
async def confirm_available_data(request: Request, req: AvailableDataConfirm):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    df = session_state.get("df_available")
    if df is None:
        raise HTTPException(status_code=400, detail="No available dataset uploaded yet.")

    selected_cols = [c for c in req.feature_cols if c in df.columns]
    if len(selected_cols) == 0:
        raise HTTPException(status_code=400, detail="Select at least 1 feature column for search.")

    expected_n_feat = get_session_n_features(session_state)

    if len(selected_cols) != expected_n_feat:
        raise HTTPException(
            status_code=400,
            detail=f"The trained model expects {expected_n_feat} feature column(s), but you selected {len(selected_cols)} feature column(s)."
        )

    BO_zone = df[selected_cols].dropna().values.astype(float)
    session_state["BO_zone_available"] = BO_zone
    SessionManager.save_session(sid, session_state)

    return {
        "status": "success",
        "num_points": len(BO_zone),
        "columns": selected_cols
    }

@app.post("/api/train-manual")
async def train_manual(request: Request, req: ManualTrainData):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    x_valid = [x for x, y in zip(req.x_points, req.y_points) if x is not None and y is not None]
    y_valid = [y for x, y in zip(req.x_points, req.y_points) if x is not None and y is not None]

    if len(x_valid) == 0:
        raise HTTPException(status_code=400, detail="Please enter at least 1 pair of training points.")

    X = np.array(x_valid, dtype=np.float64).reshape(-1, 1)
    Y = np.array(y_valid, dtype=np.float64).reshape(-1, 1)

    session_state["type_data"] = "Manual"
    session_state["Axes_titles"] = ["X", "Y"]
    session_state["Graph_title"] = "Manual Points Graph (for testing purposes)"
    session_state["ncol_data"] = 2
    session_state["n_dimensions"] = 2

    config = {
        "kernel": req.kernel,
        "norm_label": req.norm_label,
        "norm_feat": req.norm_feat,
        "likelihood": req.likelihood,
        "white_kernel": req.white_kernel,
        "split": req.split,
        "split_percentage": req.split_percentage
    }

    model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params = fit_gp_model(X, Y, config, session_state)
    SessionManager.save_session(sid, session_state)

    return {
        "status": "success",
        "n_dimensions": 2,
        "n_train": len(X_Train),
        "n_test": len(X_Test),
        "num_params": int(num_params),
        "axes_titles": ["X", "Y"],
        "var_bounds": get_var_bounds(session_state)
    }

@app.post("/api/train-csv")
async def train_csv(request: Request, req: CSVTrainData):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    df = session_state.get("df_full")
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
    Axes_titles = [n[:12] + "..." if len(n) > 12 else n for n in Col_names]

    ncol_data = df_new.shape[1]
    X = df_new.iloc[:, :-1].values.astype(float)
    Y = df_new.iloc[:, -1].values.reshape(-1, 1).astype(float)

    session_state["type_data"] = "Import"
    session_state["Axes_titles"] = Axes_titles
    session_state["ncol_data"] = int(ncol_data)
    session_state["n_dimensions"] = int(ncol_data)

    config = {
        "kernel": req.kernel,
        "norm_label": req.norm_label,
        "norm_feat": req.norm_feat,
        "likelihood": req.likelihood,
        "white_kernel": req.white_kernel,
        "split": req.split,
        "split_percentage": req.split_percentage
    }

    model, parms_X, parms_Y, X_Train, Y_Train, X_Test, Y_Test, num_params = fit_gp_model(X, Y, config, session_state)
    SessionManager.save_session(sid, session_state)

    return {
        "status": "success",
        "n_dimensions": int(ncol_data),
        "n_train": len(X_Train),
        "n_test": len(X_Test),
        "num_params": int(num_params),
        "axes_titles": Axes_titles,
        "var_bounds": get_var_bounds(session_state)
    }

@app.get("/api/parity")
async def get_parity_data(
    request: Request,
    mae: bool = False,
    mape: bool = False,
    r2: bool = False,
    rmse: bool = False,
    error_bars: bool = False
):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    if not session_state["train_done"] or session_state.get("model") is None:
        return {"train_done": False}

    model = session_state["model"]
    parms_X = session_state["parms_X"]
    parms_Y = session_state["parms_Y"]
    X_Train = session_state["X_Train"]
    Y_Train = session_state["Y_Train"]
    X_Test = session_state["X_Test"]
    Y_Test = session_state["Y_Test"]
    Y = session_state["Y"]
    var_norm_label = session_state["var_norm_label"]
    var_norm_feat = session_state["var_norm_feat"]
    Axes_titles = session_state["Axes_titles"]
    has_test = session_state["Train_Test_Split"] and X_Test is not None and len(X_Test) > 0

    X_Train_N, _ = Normalization(X_Train, var_norm_feat, parms=parms_X, reverse=False)
    Y_Train_pred_N, Y_Train_pred_std_N = model.predict_y(X_Train_N, full_cov=False)

    Y_Train_pred = Normalization(np.array(Y_Train_pred_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
    Y_Train_pred_std = Normalization(np.array(Y_Train_pred_std_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_Train_pred_N})[0]

    Y_Test_pred = None
    Y_Test_pred_std = None
    if has_test:
        X_Test_N, _ = Normalization(X_Test, var_norm_feat, parms=parms_X, reverse=False)
        Y_Test_pred_N, Y_Test_pred_std_N = model.predict_y(X_Test_N, full_cov=False)
        Y_Test_pred = Normalization(np.array(Y_Test_pred_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
        Y_Test_pred_std = Normalization(np.array(Y_Test_pred_std_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_Test_pred_N})[0]

    R2_Train = float(metrics.r2_score(Y_Train, Y_Train_pred))
    RMSE_Train = float(metrics.root_mean_squared_error(Y_Train, Y_Train_pred))
    MAE_Train = float(metrics.mean_absolute_error(Y_Train, Y_Train_pred))
    MAPE_Train = float(metrics.mean_absolute_percentage_error(Y_Train, Y_Train_pred))

    metrics_dict = {
        "R2_Train": R2_Train,
        "RMSE_Train": RMSE_Train,
        "MAE_Train": MAE_Train,
        "MAPE_Train": MAPE_Train
    }

    if has_test and Y_Test_pred is not None:
        metrics_dict.update({
            "R2_Test": float(metrics.r2_score(Y_Test, Y_Test_pred)),
            "RMSE_Test": float(metrics.root_mean_squared_error(Y_Test, Y_Test_pred)),
            "MAE_Test": float(metrics.mean_absolute_error(Y_Test, Y_Test_pred)),
            "MAPE_Test": float(metrics.mean_absolute_percentage_error(Y_Test, Y_Test_pred))
        })

    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=120)
        fig.patch.set_facecolor('#000000')
        ax.set_facecolor('#000000')

        ax.scatter(Y_Train, Y_Train_pred, s=25, facecolors='none', edgecolors='red', label="Train", zorder=3)
        if has_test and Y_Test_pred is not None:
            ax.scatter(Y_Test, Y_Test_pred, color="#3399ff", s=25, marker="x", label="Test", zorder=3)

        y_min = float(min(Y.flatten()))
        y_max = float(max(Y.flatten()))
        ax.plot((y_min, y_max), (y_min, y_max), color='#888888', linestyle='--', linewidth=1, zorder=2)

        ax.legend(fontsize=8, loc='lower right', facecolor='#1a1a1a', edgecolor='#444444')

        text_pos = 0.93
        if mae:
            ax.text(0.02, text_pos, f'MAE (Train) = {MAE_Train:.3f}', transform=ax.transAxes, color='r', fontsize=8)
            text_pos -= 0.07
        if mape:
            ax.text(0.02, text_pos, f'MAPE (Train) = {MAPE_Train:.3f}', transform=ax.transAxes, color='r', fontsize=8)
            text_pos -= 0.07
        if r2:
            ax.text(0.02, text_pos, f'R² (Train) = {R2_Train:.3f}', transform=ax.transAxes, color='r', fontsize=8)
            text_pos -= 0.07
        if rmse:
            ax.text(0.02, text_pos, f'RMSE (Train) = {RMSE_Train:.3f}', transform=ax.transAxes, color='r', fontsize=8)
            text_pos -= 0.07

        if has_test and Y_Test_pred is not None:
            MAE_Test = metrics_dict["MAE_Test"]
            MAPE_Test = metrics_dict["MAPE_Test"]
            R2_Test = metrics_dict["R2_Test"]
            RMSE_Test = metrics_dict["RMSE_Test"]
            if mae:
                ax.text(0.02, text_pos, f'MAE (Test) = {MAE_Test:.3f}', transform=ax.transAxes, color='#3399ff', fontsize=8)
                text_pos -= 0.07
            if mape:
                ax.text(0.02, text_pos, f'MAPE (Test) = {MAPE_Test:.3f}', transform=ax.transAxes, color='#3399ff', fontsize=8)
                text_pos -= 0.07
            if r2:
                ax.text(0.02, text_pos, f'R² (Test) = {R2_Test:.3f}', transform=ax.transAxes, color='#3399ff', fontsize=8)
                text_pos -= 0.07
            if rmse:
                ax.text(0.02, text_pos, f'RMSE (Test) = {RMSE_Test:.3f}', transform=ax.transAxes, color='#3399ff', fontsize=8)
                text_pos -= 0.07

        if error_bars:
            ax.errorbar(
                Y_Train.flatten(), Y_Train_pred.flatten(),
                yerr=np.sqrt(np.maximum(0, Y_Train_pred_std.flatten())),
                fmt='o', linestyle='none', capsize=3, color='#aaaaaa', ecolor='#aaaaaa', markersize=1, zorder=2
            )
            if has_test and Y_Test_pred is not None:
                ax.errorbar(
                    Y_Test.flatten(), Y_Test_pred.flatten(),
                    yerr=np.sqrt(np.maximum(0, Y_Test_pred_std.flatten())),
                    fmt='o', linestyle='none', capsize=3, color='#3399ff', ecolor='#3399ff', markersize=1, zorder=2
                )

        ax.grid(True, color='#2a2a2a', linestyle='-', linewidth=0.5)
        ax.set_title("Parity plot", color='#ffffff', fontsize=11, fontweight='bold')
        ax.set_xlabel(f"Exp. {Axes_titles[-1]}", color='#ffffff', fontsize=9)
        ax.set_ylabel(f"Pred. {Axes_titles[-1]}", color='#ffffff', fontsize=9)
        ax.tick_params(colors='#ffffff', labelsize=8)
        fig.tight_layout()

        img_b64 = fig_to_base64(fig)

    test_data = None
    if has_test and Y_Test_pred is not None:
        test_data = {
            "y_exp": Y_Test.flatten().tolist(),
            "y_pred": Y_Test_pred.flatten().tolist(),
            "y_std": np.sqrt(np.maximum(0, Y_Test_pred_std.flatten())).tolist()
        }

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
        "metrics": metrics_dict,
        "image": img_b64
    }

@app.post("/api/plot-graph")
async def get_plot_graph(request: Request, req: PlotGraphRequest):
    try:
        sid = SessionManager.get_session_id(request)
        session_state = SessionManager.get_session(sid)

        if not session_state.get("train_done") or session_state.get("model") is None:
            return {"train_done": False}

        model = session_state["model"]
        parms_X = session_state["parms_X"]
        parms_Y = session_state["parms_Y"]
        X_Train = session_state.get("X_Train")
        Y_Train = session_state.get("Y_Train")
        X_Test = session_state.get("X_Test")
        Y_Test = session_state.get("Y_Test")
        X = session_state.get("X")
        if X is None or len(X) == 0:
            X = session_state.get("X_Train")
        if X is None or len(X) == 0 or X_Train is None or Y_Train is None:
            return {"train_done": False}

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_Train = np.asarray(X_Train, dtype=np.float64)
        if X_Train.ndim == 1:
            X_Train = X_Train.reshape(-1, 1)
        Y_Train = np.asarray(Y_Train, dtype=np.float64)
        if Y_Train.ndim == 1:
            Y_Train = Y_Train.reshape(-1, 1)

        has_test = session_state.get("Train_Test_Split", False) and X_Test is not None and len(X_Test) > 0 and Y_Test is not None and len(Y_Test) > 0
        if has_test:
            X_Test = np.asarray(X_Test, dtype=np.float64)
            if X_Test.ndim == 1:
                X_Test = X_Test.reshape(-1, 1)
            Y_Test = np.asarray(Y_Test, dtype=np.float64)
            if Y_Test.ndim == 1:
                Y_Test = Y_Test.reshape(-1, 1)

        var_norm_label = session_state.get("var_norm_label", "None")
        var_norm_feat = session_state.get("var_norm_feat", "None")
        Axes_titles = session_state.get("Axes_titles", ["X", "Y"])
        Graph_title = session_state.get("Graph_title", "GRAPH")

        n_features = get_session_n_features(session_state)
        N_Points = round(int(req.n_points) ** (1 / max(1, n_features)))

        X_Plot_base = np.zeros((N_Points, n_features))

        if req.standard_plot:
            for n in range(n_features):
                varRange = np.linspace(float(X[:, n].min()), float(X[:, n].max()), N_Points)
                X_Plot_base[:, n] = varRange.copy()
        else:
            if req.var_ranges and len(req.var_ranges) >= n_features:
                Plot_min = np.array([float(r["min"]) for r in req.var_ranges[:n_features]]).reshape(-1, 1)
                Plot_max = np.array([float(r["max"]) for r in req.var_ranges[:n_features]]).reshape(-1, 1)
                Plot_Limits = np.hstack((Plot_min, Plot_max))
                for n in range(n_features):
                    varRange = np.linspace(float(Plot_Limits[n, :].min()), float(Plot_Limits[n, :].max()), N_Points)
                    X_Plot_base[:, n] = varRange.copy()
            else:
                for n in range(n_features):
                    varRange = np.linspace(float(X[:, n].min()), float(X[:, n].max()), N_Points)
                    X_Plot_base[:, n] = varRange.copy()

        X_Plot_grid = np.array(np.meshgrid(*[X_Plot_base[:, i] for i in range(n_features)])).T.reshape(-1, n_features)
        X_Plot_N, _ = Normalization(X_Plot_grid, var_norm_feat, parms=parms_X, reverse=False)

        Y_mean_N, Y_var_N = model.predict_y(X_Plot_N, full_cov=False)

        Y_mean = Normalization(np.array(Y_mean_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True)[0]
        Y_var = Normalization(np.array(Y_var_N).reshape(-1, 1), var_norm_label, parms=parms_Y, reverse=True, var={"bol": True, "Y_N": Y_mean_N})[0]

        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(6, 4.2), dpi=120)
            fig.patch.set_facecolor('#000000')

            if n_features == 1:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#000000')
                Y_Upper = Y_mean.flatten() + 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))
                Y_Lower = Y_mean.flatten() - 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))

                m_color = getattr(req, "model_color", "black")
                if m_color == "black":
                    m_color = "#ffffff"
                ic_color = getattr(req, "ic_color", "blue")
                if ic_color == "blue":
                    ic_color = "#3399ff"

                ax.plot(X_Plot_grid[:, 0], Y_mean.flatten(), label="Y mean", color=m_color, linewidth=2)
                ax.plot(X_Plot_grid[:, 0], Y_Upper, "--", label="Y I.C. 95%", color=ic_color, linewidth=1.5)
                ax.plot(X_Plot_grid[:, 0], Y_Lower, "--", color=ic_color, linewidth=1.5)
                ax.fill_between(X_Plot_grid[:, 0], Y_Lower, Y_Upper, color=ic_color, alpha=0.15)

                ax.plot(X_Train[:, 0], Y_Train.flatten(), "o", label="Train", color="red", markersize=5)
                if has_test and X_Test is not None and Y_Test is not None:
                    ax.plot(X_Test[:, 0], Y_Test.flatten(), "o", label="Test", color="#3399ff", markersize=5)

                ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444')
                ax.grid(True, color='#2a2a2a')
                ax.set_title(Graph_title, color='#ffffff', fontsize=11, fontweight='bold')
                ax.set_xlabel(Axes_titles[0] if len(Axes_titles) > 0 else "X", color='#ffffff', fontsize=9)
                ax.set_ylabel(Axes_titles[1] if len(Axes_titles) > 1 else "Y", color='#ffffff', fontsize=9)
                ax.tick_params(colors='#ffffff', labelsize=8)

            elif n_features == 2:
                ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
                ax.set_facecolor('#000000')
                cmap_colour = get_safe_cmap(getattr(req, "cmap", "viridis"))

                try:
                    ax.plot_trisurf(X_Plot_grid[:, 0], X_Plot_grid[:, 1], Y_mean.flatten(), cmap=cmap_colour)
                except Exception:
                    ax.scatter(X_Plot_grid[:, 0], X_Plot_grid[:, 1], Y_mean.flatten(), c=Y_mean.flatten(), cmap=cmap_colour, s=15)

                ax.view_init(elev=15, azim=310)

                ax.plot(X_Train[:, 0], X_Train[:, 1], Y_Train.flatten(), "o", color="red", zorder=4.6, markersize=4, label="Train")
                if has_test and X_Test is not None and Y_Test is not None:
                    ax.plot(X_Test[:, 0], X_Test[:, 1], Y_Test.flatten(), "o", color="#3399ff", zorder=4.6, markersize=4, label="Test")

                ax.set_title(Graph_title, color='#ffffff', fontsize=11, fontweight='bold')
                ax.set_xlabel(Axes_titles[0] if len(Axes_titles) > 0 else "X1", color='#ffffff', fontsize=8)
                ax.set_ylabel(Axes_titles[1] if len(Axes_titles) > 1 else "X2", color='#ffffff', fontsize=8)
                ax.set_zlabel(Axes_titles[2] if len(Axes_titles) > 2 else "Y", color='#ffffff', fontsize=8)
                ax.tick_params(colors='#ffffff', labelsize=7)
            else:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#000000')
                ax.set_title("GRAPH", color='#ffffff')
                ax.set_xlabel("X", color='#ffffff')
                ax.set_ylabel("Y", color='#ffffff')

            fig.tight_layout()
            img_b64 = fig_to_base64(fig)

        result = {
            "train_done": True,
            "n_features": n_features,
            "axes_titles": Axes_titles,
            "graph_title": Graph_title,
            "has_test": has_test,
            "image": img_b64
        }

        if n_features == 1:
            result.update({
                "x_plot": X_Plot_grid[:, 0].tolist(),
                "y_mean": Y_mean.flatten().tolist(),
                "y_upper": (Y_mean.flatten() + 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))).tolist(),
                "y_lower": (Y_mean.flatten() - 1.96 * np.sqrt(np.maximum(0, Y_var.flatten()))).tolist()
            })
        elif n_features == 2:
            x1_axis = X_Plot_base[:, 0]
            x2_axis = X_Plot_base[:, 1]
            Z_grid = Y_mean.reshape(N_Points, N_Points)
            result.update({
                "x1_axis": x1_axis.tolist(),
                "x2_axis": x2_axis.tolist(),
                "z_surface": Z_grid.tolist()
            })

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Plot Error: {str(e)}")

@app.post("/api/predict-y")
async def predict_y(request: Request, req: PredictYRequest):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    if not session_state["train_done"] or session_state.get("model") is None:
        raise HTTPException(status_code=400, detail="Model is not trained yet.")

    model = session_state["model"]
    parms_X = session_state["parms_X"]
    parms_Y = session_state["parms_Y"]
    var_norm_label = session_state["var_norm_label"]
    var_norm_feat = session_state["var_norm_feat"]

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
async def run_albo(request: Request, req: ALBORequest):
    try:
        sid = SessionManager.get_session_id(request)
        session_state = SessionManager.get_session(sid)

        if not session_state.get("train_done") or session_state.get("model") is None:
            raise HTTPException(status_code=400, detail="Model is not trained yet.")

        model = session_state["model"]
        parms_X = session_state["parms_X"]
        parms_Y = session_state["parms_Y"]
        X_Train = session_state.get("X_Train")
        Y_Train = session_state.get("Y_Train")
        X_Test = session_state.get("X_Test")
        Y_Test = session_state.get("Y_Test")
        X = session_state.get("X")
        if X is None or len(X) == 0:
            X = session_state.get("X_Train")

        if X_Train is not None:
            X_Train = np.asarray(X_Train, dtype=np.float64)
            if X_Train.ndim == 1:
                X_Train = X_Train.reshape(-1, 1)
        if Y_Train is not None:
            Y_Train = np.asarray(Y_Train, dtype=np.float64)
            if Y_Train.ndim == 1:
                Y_Train = Y_Train.reshape(-1, 1)

        has_test = session_state.get("Train_Test_Split", False) and X_Test is not None and len(X_Test) > 0 and Y_Test is not None and len(Y_Test) > 0
        if has_test:
            X_Test = np.asarray(X_Test, dtype=np.float64)
            if X_Test.ndim == 1:
                X_Test = X_Test.reshape(-1, 1)
            Y_Test = np.asarray(Y_Test, dtype=np.float64)
            if Y_Test.ndim == 1:
                Y_Test = Y_Test.reshape(-1, 1)

        n_features = get_session_n_features(session_state)

        var_norm_label = session_state.get("var_norm_label", "None")
        var_norm_feat = session_state.get("var_norm_feat", "None")
        Axes_titles = session_state.get("Axes_titles", ["X", "Y"])
        Graph_title = session_state.get("Graph_title", "GRAPH")

        BO_zone = None

        if req.import_available:
            if session_state.get("BO_zone_available") is not None:
                cand = np.asarray(session_state["BO_zone_available"], dtype=np.float64)
                if cand.ndim == 1:
                    cand = cand.reshape(-1, 1)
                if cand.shape[1] == n_features:
                    BO_zone = cand
                else:
                    session_state.pop("BO_zone_available", None)

            if BO_zone is None:
                if session_state.get("df_available") is not None:
                    df = session_state["df_available"]
                    if df.shape[1] >= n_features:
                        BO_zone = df.iloc[:, :n_features].values.astype(float)
                elif session_state.get("df_full") is not None:
                    df = session_state["df_full"]
                    if df.shape[1] >= n_features:
                        BO_zone = df.iloc[:, :n_features].values.astype(float)

            if BO_zone is None or BO_zone.shape[1] != n_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"The trained model expects {n_features} feature column(s), but no matching search dataset was provided. Please click 'Available Data' to select matching candidate columns."
                )
        else:
            if X is None or len(X) == 0:
                raise HTTPException(status_code=400, detail="Training data not found for domain limits calculation.")
            X = np.asarray(X, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(-1, 1)

            if req.standard_plot:
                x_min = np.array([float(np.min(X[:, e])) for e in range(n_features)]).reshape(-1, 1)
                x_max = np.array([float(np.max(X[:, e])) for e in range(n_features)]).reshape(-1, 1)
            else:
                if req.x_ranges and len(req.x_ranges) >= n_features:
                    x_min = np.array([float(req.x_ranges[e]["min"]) for e in range(n_features)]).reshape(-1, 1)
                    x_max = np.array([float(req.x_ranges[e]["max"]) for e in range(n_features)]).reshape(-1, 1)
                else:
                    x_min = np.array([float(np.min(X[:, e])) for e in range(n_features)]).reshape(-1, 1)
                    x_max = np.array([float(np.max(X[:, e])) for e in range(n_features)]).reshape(-1, 1)

            N_Points = round(int(req.n_points) ** (1 / max(1, n_features)))
            BO_zone_base = np.zeros((N_Points, n_features))

            for n in range(n_features):
                varRange = np.linspace(float(x_min[n]), float(x_max[n]), N_Points)
                BO_zone_base[:, n] = varRange.copy()

            BO_zone = np.array(np.meshgrid(*[BO_zone_base[:, i] for i in range(n_features)])).T.reshape(-1, n_features)

        BO_zone = np.asarray(BO_zone, dtype=np.float64)
        if BO_zone.ndim == 1:
            BO_zone = BO_zone.reshape(-1, 1)

        if BO_zone.shape[1] != n_features:
            raise HTTPException(
                status_code=400,
                detail=f"The trained model expects {n_features} feature column(s), but the search dataset has {BO_zone.shape[1]} column(s). Please confirm available data columns."
            )

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

        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(6, 4.2), dpi=120)
            fig.patch.set_facecolor('#000000')

            if n_features == 1:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#000000')

                m_color = getattr(req, "model_color", "black")
                if m_color == "black":
                    m_color = "#ffffff"
                af_color = getattr(req, "af_color", "blue")
                if af_color == "blue":
                    af_color = "#3399ff"

                ax.plot(BO_zone[:, 0], Y_mean_flat, label="Y mean", color=m_color, linewidth=2)
                ax.plot(BO_zone[:, 0], AF, "--", label="AF", color=af_color, linewidth=1.5)
                if X_Train is not None and Y_Train is not None:
                    ax.plot(X_Train[:, 0], Y_Train.flatten(), "o", label="Train", color="red", markersize=5)
                if has_test and X_Test is not None and Y_Test is not None:
                    ax.plot(X_Test[:, 0], Y_Test.flatten(), "o", label="Test", color="#3399ff", markersize=5)

                ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444444')
                ax.grid(True, color='#2a2a2a')
                ax.set_title(Graph_title, color='#ffffff', fontsize=11, fontweight='bold')
                ax.set_xlabel(Axes_titles[0] if len(Axes_titles) > 0 else "X", color='#ffffff', fontsize=9)
                ax.set_ylabel("A.F.", color='#ffffff', fontsize=9)
                ax.tick_params(colors='#ffffff', labelsize=8)

            elif n_features == 2:
                ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
                ax.set_facecolor('#000000')
                cmap_colour = get_safe_cmap(getattr(req, "cmap", "viridis"))

                try:
                    ax.plot_trisurf(BO_zone[:, 0], BO_zone[:, 1], AF.reshape(-1,), cmap=cmap_colour)
                except Exception:
                    ax.scatter(BO_zone[:, 0], BO_zone[:, 1], AF.reshape(-1,), c=AF.reshape(-1,), cmap=cmap_colour, s=20)

                ax.view_init(elev=15, azim=310)

                if X_Train is not None and Y_Train is not None:
                    ax.plot(X_Train[:, 0], X_Train[:, 1], Y_Train.flatten(), "o", color="red", zorder=4.6, markersize=4, label="Train")
                if has_test and X_Test is not None and Y_Test is not None:
                    ax.plot(X_Test[:, 0], X_Test[:, 1], Y_Test.flatten(), "o", color="#3399ff", zorder=4.6, markersize=4, label="Test")

                ax.set_title(Graph_title, color='#ffffff', fontsize=11, fontweight='bold')
                ax.set_xlabel(Axes_titles[0] if len(Axes_titles) > 0 else "X1", color='#ffffff', fontsize=8)
                ax.set_ylabel(Axes_titles[1] if len(Axes_titles) > 1 else "X2", color='#ffffff', fontsize=8)
                ax.set_zlabel("A.F.", color='#ffffff', fontsize=8)
                ax.tick_params(colors='#ffffff', labelsize=7)
            else:
                ax = fig.add_subplot(111)
                ax.set_facecolor('#000000')
                ax.set_title("GRAPH", color='#ffffff')
                ax.set_xlabel("X", color='#ffffff')
                ax.set_ylabel("Y", color='#ffffff')

            fig.tight_layout()
            img_b64 = fig_to_base64(fig)

        result = {
            "af_type": var_AF,
            "max_af": round(max_AF_val, 4),
            "next_point": [round(p, 3) for p in next_point],
            "n_features": n_features,
            "axes_titles": Axes_titles,
            "graph_title": Graph_title,
            "has_test": has_test,
            "image": img_b64
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AL/BO Error: {str(e)}")

def build_session_variables(session_state: dict[str, Any]):
    def to_list(val):
        if isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, pd.DataFrame):
            return val.to_dict(orient="records")
        return val

    return {
        "train_done": session_state.get("train_done", False),
        "parms_X": to_list(session_state.get("parms_X")),
        "parms_Y": to_list(session_state.get("parms_Y")),
        "X_Train": to_list(session_state.get("X_Train")),
        "Y_Train": to_list(session_state.get("Y_Train")),
        "X_Test": to_list(session_state.get("X_Test")),
        "Y_Test": to_list(session_state.get("Y_Test")),
        "ncol_data": session_state.get("ncol_data", 2),
        "X": to_list(session_state.get("X")),
        "Y": to_list(session_state.get("Y")),
        "Axes_titles": session_state.get("Axes_titles", ["X", "Y"]),
        "Graph_title": session_state.get("Graph_title", ""),
        "var_norm_label": session_state.get("var_norm_label", "None"),
        "var_norm_feat": session_state.get("var_norm_feat", "None"),
        "var_kernel": session_state.get("var_kernel", "RBF"),
        "trainlikelihood": session_state.get("trainlikelihood", False),
        "white_kernel": session_state.get("white_kernel", False),
        "Train_Test_Split": session_state.get("Train_Test_Split", False),
        "Split_Percentage": session_state.get("Split_Percentage", 20.0),
        "num_params": session_state.get("num_params", 0),
        "type_data": session_state.get("type_data", "Manual"),
        "df_full": to_list(session_state.get("df_full")),
        "df_available": to_list(session_state.get("df_available")),
        "BO_zone_available": to_list(session_state.get("BO_zone_available")),
        "Save_path": session_state.get("Save_path")
    }

@app.api_route("/api/save-as", methods=["GET", "POST"])
async def save_as(request: Request, filename: str | None = None):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    if not filename:
        filename = "gp_session.json"
    if not filename.endswith(".json"):
        if filename.endswith(".pkl"):
            filename = filename[:-4] + ".json"
        else:
            filename += ".json"

    session_state["Save_path"] = filename
    SessionManager.save_session(sid, session_state)
    variables = build_session_variables(session_state)

    json_data = json.dumps(variables, indent=2)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@app.api_route("/api/save", methods=["GET", "POST"])
async def save(request: Request):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    save_path = session_state.get("Save_path")
    if not save_path:
        return JSONResponse(content={"status": "no_save_path"}, status_code=200)

    variables = build_session_variables(session_state)
    json_data = json.dumps(variables, indent=2)
    return Response(
        content=json_data,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{save_path}"'}
    )

@app.get("/api/export-session")
async def export_session(request: Request):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)
    return await save_as(request, session_state.get("Save_path") or "gp_session.json")

@app.post("/api/load-session")
async def load_session(request: Request, file: UploadFile = File(...)):
    sid = SessionManager.get_session_id(request)
    session_state = SessionManager.get_session(sid)

    contents = await file.read()
    filename = file.filename or "loaded_session.json"

    try:
        text = contents.decode("utf-8-sig", errors="ignore")
        data = json.loads(text)
        validated = SessionSaveSchema(**data)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid session file format (.json expected with valid schema): {e!s}"
        )

    loaded_dict = validated.model_dump()
    for key, val in loaded_dict.items():
        if val is not None:
            session_state[key] = val

    session_state["Save_path"] = filename

    # Re-fit GP model if training data was present in loaded session
    if session_state.get("train_done") and session_state.get("X") is not None and session_state.get("Y") is not None:
        X = np.array(session_state["X"], dtype=np.float64)
        Y = np.array(session_state["Y"], dtype=np.float64)
        config = {
            "kernel": session_state.get("var_kernel", "RBF"),
            "norm_label": session_state.get("var_norm_label", "None"),
            "norm_feat": session_state.get("var_norm_feat", "None"),
            "likelihood": session_state.get("trainlikelihood", False),
            "white_kernel": session_state.get("white_kernel", False),
            "split": session_state.get("Train_Test_Split", False),
            "split_percentage": session_state.get("Split_Percentage", 20.0)
        }
        try:
            fit_gp_model(X, Y, config, session_state)
        except Exception as e:
            print(f"Warning: Re-fitting GP model during session load failed: {e}")

    SessionManager.save_session(sid, session_state)

    def to_list(arr):
        return arr.tolist() if isinstance(arr, np.ndarray) else arr

    n_train = len(session_state["X_Train"]) if session_state.get("X_Train") is not None else 0
    n_test = len(session_state["X_Test"]) if session_state.get("X_Test") is not None else 0

    return {
        "status": "success",
        "Save_path": filename,
        "session_data": {
            "type_data": session_state.get("type_data", "Manual"),
            "train_done": session_state.get("train_done", False),
            "var_kernel": session_state.get("var_kernel", "RBF"),
            "var_norm_label": session_state.get("var_norm_label", "None"),
            "var_norm_feat": session_state.get("var_norm_feat", "None"),
            "trainlikelihood": session_state.get("trainlikelihood", False),
            "white_kernel": session_state.get("white_kernel", False),
            "Train_Test_Split": session_state.get("Train_Test_Split", False),
            "Split_Percentage": session_state.get("Split_Percentage", 20.0),
            "ncol_data": session_state.get("ncol_data", 2),
            "Axes_titles": session_state.get("Axes_titles", ["X", "Y"]),
            "Graph_title": session_state.get("Graph_title", ""),
            "num_params": session_state.get("num_params", 0),
            "X": to_list(session_state.get("X")),
            "Y": to_list(session_state.get("Y")),
            "X_Train": to_list(session_state.get("X_Train")),
            "Y_Train": to_list(session_state.get("Y_Train")),
            "X_Test": to_list(session_state.get("X_Test")),
            "Y_Test": to_list(session_state.get("Y_Test")),
        },
        "n_train": n_train,
        "n_test": n_test,
        "num_params": session_state.get("num_params", 0),
        "var_bounds": get_var_bounds(session_state)
    }

@app.post("/api/global-reset")
async def global_reset(request: Request):
    sid = SessionManager.get_session_id(request)
    session_state = get_default_session_state()
    SessionManager.save_session(sid, session_state)
    SESSION_MODELS.pop(sid, None)
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
        return FileResponse(index_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return HTMLResponse(content="<h1>GP Training App Web</h1><p>Index file missing.</p>")

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    uvicorn.run(app, host=host, port=port, log_level=log_level)
