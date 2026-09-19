// Session ID Management for Multi-User Isolated Sessions
function getSessionId() {
    let sid = localStorage.getItem("gp_session_id");
    if (!sid) {
        sid = (typeof crypto !== "undefined" && crypto.randomUUID) ? crypto.randomUUID() : "session-" + Math.random().toString(36).substring(2) + Date.now().toString(36);
        localStorage.setItem("gp_session_id", sid);
    }
    return sid;
}

// Wrapper around fetch to automatically inject X-Session-ID header
async function apiFetch(url, options = {}) {
    options = options || {};
    options.headers = options.headers || {};
    if (options.headers instanceof Headers) {
        options.headers.set("X-Session-ID", getSessionId());
    } else {
        options.headers["X-Session-ID"] = getSessionId();
    }
    return fetch(url, options);
}

// State Variables
let currentSession = {
    train_done: false,
    type_data: "Manual",
    n_dimensions: 2,
    axes_titles: ["X", "Y"],
    var_bounds: [{ name: "X", min: 0, max: 10 }],
    csv_columns: [],
    available_data_columns: []
};

// Dark theme layout configuration for Plotly
const plotlyDarkLayout = {
    autosize: true,
    paper_bgcolor: "#000000",
    plot_bgcolor: "#000000",
    font: { family: "Calibri, Segoe UI, sans-serif", color: "#ffffff", size: 11 },
    margin: { l: 45, r: 15, t: 30, b: 35, autoexpand: true },
    xaxis: { gridcolor: "#2a2a2a", zerolinecolor: "#444444" },
    yaxis: { gridcolor: "#2a2a2a", zerolinecolor: "#444444" }
};

const plotlyConfig = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['lasso2d', 'select2d']
};

// Multi-window & Modal Management System
let highestZIndex = 2000;

function bringToFront(modalEl) {
    if (!modalEl) return;
    highestZIndex++;
    modalEl.style.zIndex = highestZIndex;
    const win = modalEl.querySelector('.modal-window');
    if (win) win.style.zIndex = highestZIndex;
}

const defaultModalPositions = {
    modalPlot: { top: 60, left: 80 },
    modalPred: { top: 90, left: 520 },
    modalALBO: { top: 110, left: 200 },
    modalCSVVars: { top: 120, left: 350 },
    modalAvailableVars: { top: 140, left: 380 },
    modalSaveAs: { top: 200, left: 400 }
};

function openModal(id) {
    const modal = document.getElementById(id);
    if (!modal) return;
    
    modal.style.display = "flex";
    bringToFront(modal);
    
    const win = modal.querySelector('.modal-window');
    if (win) {
        if (!win.style.top || !win.style.left) {
            const pos = defaultModalPositions[id] || { top: 100, left: 200 };
            win.style.top = `${pos.top}px`;
            win.style.left = `${pos.left}px`;
        }
    }
    
    // Trigger rendering after container is visible
    setTimeout(() => {
        if (id === 'modalPlot') {
            renderGPPlotLimits();
            renderGPPlot();
        } else if (id === 'modalALBO') {
            renderALBOLimits();
            renderAFPlot();
        } else if (id === 'modalPred') {
            renderPredictInputs();
        }
    }, 60);
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.style.display = "none";
}

function makeWindowDraggable(modalEl) {
    const header = modalEl.querySelector('.modal-header');
    const win = modalEl.querySelector('.modal-window') || modalEl;
    if (!header || !win) return;

    win.addEventListener('mousedown', () => {
        bringToFront(modalEl);
    });

    let isDragging = false;
    let startX = 0, startY = 0;
    let initialLeft = 0, initialTop = 0;

    header.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('modal-close')) return;

        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;

        const rect = win.getBoundingClientRect();
        initialLeft = rect.left;
        initialTop = rect.top;

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);

        e.preventDefault();
    });

    function onMouseMove(e) {
        if (!isDragging) return;
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        win.style.left = `${initialLeft + dx}px`;
        win.style.top = `${initialTop + dy}px`;
    }

    function onMouseUp() {
        isDragging = false;
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);
    }
}

// Split Checkbox toggle
function toggleSplitInput() {
    const chk = document.getElementById("chkSplit");
    const txt = document.getElementById("txtTestSplit");
    txt.disabled = !chk.checked;
}

// Standard plot toggles
function toggleGPPlotStandard() {
    const isStd = document.getElementById("chkPlotStandard").checked;
    document.querySelectorAll(".gp-min, .gp-max").forEach(input => {
        input.disabled = isStd;
    });
}

function toggleALBOStandard() {
    const isStd = document.getElementById("chkALBOStandard").checked;
    document.querySelectorAll(".albo-min, .albo-max").forEach(input => {
        input.disabled = isStd;
    });
}

function toggleALBOAvailable() {
    const isAvail = document.getElementById("chkALBOAvailable").checked;
    if (isAvail) {
        document.getElementById("chkALBOStandard").disabled = true;
    } else {
        document.getElementById("chkALBOStandard").disabled = false;
    }
}

// Handle Main CSV Upload
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await apiFetch("/api/upload-csv", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to parse CSV file");

        currentSession.csv_columns = data.columns;
        currentSession.type_data = "Import";

        // Disable manual points table inputs as in original Open_CSV_File()
        document.querySelectorAll(".points-table input").forEach(input => {
            input.disabled = true;
        });

        populateCSVModal(data.columns);
    } catch (err) {
        alert("Error uploading CSV: " + err.message);
    } finally {
        event.target.value = "";
    }
}

function populateCSVModal(columns) {
    const selLabel = document.getElementById("selCSVLabel");
    const featList = document.getElementById("csvFeaturesList");

    selLabel.innerHTML = "";
    featList.innerHTML = "";

    columns.forEach(col => {
        const opt = document.createElement("option");
        opt.value = col;
        opt.textContent = col;
        selLabel.appendChild(opt);
    });
    selLabel.selectedIndex = columns.length - 1;

    columns.forEach(col => {
        const div = document.createElement("div");
        div.className = "form-row";
        div.style.justifyContent = "flex-start";
        div.style.gap = "8px";
        div.innerHTML = `<label class="checkbox-label"><input type="checkbox" class="csv-feat-chk" value="${col}" checked> ${col}</label>`;
        featList.appendChild(div);
    });
}

function toggleSelectAllCSVFeatures() {
    const chkAll = document.getElementById("chkCSVSelectAll").checked;
    document.querySelectorAll(".csv-feat-chk").forEach(chk => chk.checked = chkAll);
}

// Automatic Training & Parity Plot after CSV confirmation
async function confirmCSVVariables() {
    const labelCol = document.getElementById("selCSVLabel").value;
    const featCols = Array.from(document.querySelectorAll(".csv-feat-chk:checked"))
        .map(c => c.value)
        .filter(c => c !== labelCol);

    if (featCols.length === 0) {
        alert("Please select at least 1 feature column.");
        return;
    }

    currentSession.type_data = "Import";
    closeModal("modalCSVVars");
    await trainModelCSV(labelCol, featCols);
}

// Handle Available Data CSV Upload for AL/BO
async function handleAvailableDataUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await apiFetch("/api/upload-available-data", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to parse Available Data CSV");

        currentSession.available_data_columns = data.columns;
        populateAvailableDataModal(data.columns);
        openModal("modalAvailableVars");
    } catch (err) {
        alert("Error uploading Available Data: " + err.message);
    } finally {
        event.target.value = "";
    }
}

function populateAvailableDataModal(columns) {
    const list = document.getElementById("availableFeaturesList");
    list.innerHTML = "";

    columns.forEach(col => {
        const div = document.createElement("div");
        div.className = "form-row";
        div.style.justifyContent = "flex-start";
        div.style.gap = "8px";
        div.innerHTML = `<label class="checkbox-label"><input type="checkbox" class="avail-feat-chk" value="${col}" checked> ${col}</label>`;
        list.appendChild(div);
    });
}

async function confirmAvailableDataVariables() {
    const featCols = Array.from(document.querySelectorAll(".avail-feat-chk:checked")).map(c => c.value);
    if (featCols.length === 0) {
        alert("Please select at least 1 feature column.");
        return;
    }

    try {
        const res = await apiFetch("/api/confirm-available-data", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ feature_cols: featCols })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to confirm available data");

        closeModal("modalAvailableVars");
        alert(`Available dataset loaded with ${data.num_points} candidate points.`);
        await renderAFPlot();
    } catch (err) {
        alert("Error: " + err.message);
    }
}

// Train Model (Manual / CSV Router)
async function trainModel() {
    if (currentSession.type_data === "Import") {
        if (!currentSession.csv_columns || currentSession.csv_columns.length === 0) {
            alert("No CSV file loaded. Please upload a CSV file first.");
            return;
        }
        openModal("modalCSVVars");
        return;
    }

    const xInputs = document.querySelectorAll(".manual-x");
    const yInputs = document.querySelectorAll(".manual-y");

    const xPts = Array.from(xInputs).map(i => parseFloat(i.value)).filter(val => !isNaN(val));
    const yPts = Array.from(yInputs).map(i => parseFloat(i.value)).filter(val => !isNaN(val));

    const payload = {
        x_points: xPts,
        y_points: yPts,
        kernel: document.getElementById("selKernel").value,
        norm_label: document.getElementById("selNormLabel").value,
        norm_feat: document.getElementById("selNormFeat").value,
        likelihood: document.getElementById("chkLikelihood").checked,
        white_kernel: document.getElementById("chkWhiteKernel").checked,
        split: document.getElementById("chkSplit").checked,
        split_percentage: parseFloat(document.getElementById("txtTestSplit").value) || 20.0
    };

    try {
        const res = await apiFetch("/api/train-manual", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Training failed");

        updateModelDetails(data);
        await updateParityPlot();
    } catch (err) {
        alert("Training Error: " + err.message);
    }
}

// Train Model (CSV)
async function trainModelCSV(labelCol, featCols) {
    const payload = {
        label_col: labelCol,
        feature_cols: featCols,
        kernel: document.getElementById("selKernel").value,
        norm_label: document.getElementById("selNormLabel").value,
        norm_feat: document.getElementById("selNormFeat").value,
        likelihood: document.getElementById("chkLikelihood").checked,
        white_kernel: document.getElementById("chkWhiteKernel").checked,
        split: document.getElementById("chkSplit").checked,
        split_percentage: parseFloat(document.getElementById("txtTestSplit").value) || 20.0
    };

    try {
        const res = await apiFetch("/api/train-csv", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Training failed");

        updateModelDetails(data);
        await updateParityPlot();
    } catch (err) {
        alert("Training Error: " + err.message);
    }
}

function updateModelDetails(data) {
    currentSession.train_done = true;
    currentSession.n_dimensions = data.n_dimensions;
    currentSession.axes_titles = data.axes_titles;
    currentSession.var_bounds = data.var_bounds || [];

    document.getElementById("lblDimensions").textContent = data.n_dimensions;
    document.getElementById("lblTrainPoints").textContent = data.n_train;
    document.getElementById("lblTestPoints").textContent = data.n_test;
    document.getElementById("lblHyperparams").textContent = data.num_params;

    // Lock options post-train (matching original Tkinter behavior)
    document.getElementById("selKernel").disabled = true;
    document.getElementById("selNormLabel").disabled = true;
    document.getElementById("selNormFeat").disabled = true;
    document.getElementById("chkLikelihood").disabled = true;
    document.getElementById("chkWhiteKernel").disabled = true;
    document.getElementById("chkSplit").disabled = true;
    document.getElementById("btnTrainModel").disabled = true;
    document.querySelectorAll(".manual-x, .manual-y").forEach(inp => inp.disabled = true);

    // Update dynamic variable inputs across all modules
    updateAllVariableInputs();
}

// Update Dynamic Inputs for Predict Y, GP Plot, and ALBO with visible labels on TOP
function updateAllVariableInputs() {
    renderPredictInputs();
    renderGPPlotLimits();
    renderALBOLimits();
}

function renderPredictInputs() {
    const list = document.getElementById("predInputsList");
    if (!list) return;
    list.innerHTML = "";

    const nFeat = Math.max(1, currentSession.n_dimensions - 1);
    const titles = currentSession.axes_titles;
    const labelTitle = titles[titles.length - 1] || "Y";
    document.getElementById("lblPredYTitle").textContent = `Pred. ${labelTitle} :`;

    for (let i = 0; i < nFeat; i++) {
        const vName = titles[i] || `Variable ${i + 1}`;
        const container = document.createElement("div");
        container.className = "var-column-input";
        container.innerHTML = `
            <div class="var-top-label" title="${vName}">${vName}</div>
            <input type="number" step="any" class="pred-x-val" style="width: 70px; text-align: center;" value="0">
        `;
        list.appendChild(container);
    }
}

function renderGPPlotLimits() {
    const container = document.getElementById("gpPlotLimitsContainer");
    if (!container) return;
    container.innerHTML = "";

    const nFeat = Math.max(1, currentSession.n_dimensions - 1);
    const titles = currentSession.axes_titles;
    const bounds = currentSession.var_bounds;
    const isStd = document.getElementById("chkPlotStandard").checked;

    for (let i = 0; i < nFeat; i++) {
        const vName = titles[i] || `Variable ${i + 1}`;
        const b = bounds[i] || { min: 0, max: 10 };
        const row = document.createElement("div");
        row.className = "var-limits-row";
        row.innerHTML = `
            <div class="var-limits-label" title="${vName}">${vName} :</div>
            <div class="var-limits-inputs">
                <input type="number" step="any" class="gp-min" style="width: 55px; text-align: center;" value="${b.min}" ${isStd ? 'disabled' : ''}>
                <input type="number" step="any" class="gp-max" style="width: 55px; text-align: center;" value="${b.max}" ${isStd ? 'disabled' : ''}>
            </div>
        `;
        container.appendChild(row);
    }
}

function renderALBOLimits() {
    const container = document.getElementById("alboLimitsContainer");
    if (!container) return;
    container.innerHTML = "";

    const nFeat = Math.max(1, currentSession.n_dimensions - 1);
    const titles = currentSession.axes_titles;
    const bounds = currentSession.var_bounds;
    const isStd = document.getElementById("chkALBOStandard").checked;

    for (let i = 0; i < nFeat; i++) {
        const vName = titles[i] || `Variable ${i + 1}`;
        const b = bounds[i] || { min: 0, max: 10 };
        const row = document.createElement("div");
        row.className = "albo-grid-row";
        row.innerHTML = `
            <span class="var-limits-label" title="${vName}">${vName} :</span>
            <input type="number" step="any" class="albo-min" style="width: 55px; text-align: center;" value="${b.min}" ${isStd ? 'disabled' : ''}>
            <input type="number" step="any" class="albo-max" style="width: 55px; text-align: center;" value="${b.max}" ${isStd ? 'disabled' : ''}>
            <span class="albo-result-val" style="font-weight: bold; color: #00ffcc; text-align: center;">---</span>
        `;
        container.appendChild(row);
    }
}

// Parity Plot Update - Purely Reactive Plotly
async function updateParityPlot() {
    const plotDiv = document.getElementById("parityPlotDiv");
    if (!plotDiv) return;

    try {
        const res = await apiFetch("/api/parity");
        const data = await res.json();

        if (!data.train_done) {
            Plotly.newPlot("parityPlotDiv", [], {
                ...plotlyDarkLayout,
                title: "Parity plot"
            }, plotlyConfig);
            setTimeout(() => Plotly.Plots.resize("parityPlotDiv"), 50);
            return;
        }

        const traces = [];
        const labelY = data.axes_titles[data.axes_titles.length - 1];

        // Train scatter trace
        const trainTrace = {
            x: data.train.y_exp,
            y: data.train.y_pred,
            mode: "markers",
            name: "Train",
            marker: { color: "red", size: 6, symbol: "circle-open", line: { width: 1.5 } }
        };

        if (document.getElementById("chkErrorBars").checked && data.train.y_std) {
            trainTrace.error_y = {
                type: "data",
                array: data.train.y_std,
                visible: true,
                color: "#ffffff"
            };
        }
        traces.push(trainTrace);

        // Test scatter trace
        if (data.has_test && data.test) {
            const testTrace = {
                x: data.test.y_exp,
                y: data.test.y_pred,
                mode: "markers",
                name: "Test",
                marker: { color: "#3399ff", size: 6, symbol: "x" }
            };

            if (document.getElementById("chkErrorBars").checked && data.test.y_std) {
                testTrace.error_y = {
                    type: "data",
                    array: data.test.y_std,
                    visible: true,
                    color: "#ffffff"
                };
            }
            traces.push(testTrace);
        }

        // 1:1 Reference Line
        traces.push({
            x: [data.line_min, data.line_max],
            y: [data.line_min, data.line_max],
            mode: "lines",
            name: "1:1 Line",
            line: { color: "#888888", dash: "dash", width: 1 }
        });

        // Annotations for metrics
        const annotations = [];
        let yPos = 0.95;

        const m = data.metrics;
        if (m) {
            if (document.getElementById("chkMAE").checked && m.MAE_Train !== undefined) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Train) = ${m.MAE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkMAPE").checked && m.MAPE_Train !== undefined) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Train) = ${m.MAPE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkR2").checked && m.R2_Train !== undefined) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Train) = ${m.R2_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkRMSE").checked && m.RMSE_Train !== undefined) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Train) = ${m.RMSE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
                yPos -= 0.07;
            }

            if (data.has_test) {
                if (document.getElementById("chkMAE").checked && m.MAE_Test !== undefined) {
                    annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Test) = ${m.MAE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                    yPos -= 0.07;
                }
                if (document.getElementById("chkMAPE").checked && m.MAPE_Test !== undefined) {
                    annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Test) = ${m.MAPE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                    yPos -= 0.07;
                }
                if (document.getElementById("chkR2").checked && m.R2_Test !== undefined) {
                    annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Test) = ${m.R2_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                    yPos -= 0.07;
                }
                if (document.getElementById("chkRMSE").checked && m.RMSE_Test !== undefined) {
                    annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Test) = ${m.RMSE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                    yPos -= 0.07;
                }
            }
        }

        const layout = {
            ...plotlyDarkLayout,
            title: "Parity plot",
            xaxis: { ...plotlyDarkLayout.xaxis, title: `Exp. ${labelY}` },
            yaxis: { ...plotlyDarkLayout.yaxis, title: `Pred. ${labelY}` },
            annotations: annotations
        };

        Plotly.newPlot("parityPlotDiv", traces, layout, plotlyConfig);
        Plotly.Plots.resize("parityPlotDiv");
        setTimeout(() => Plotly.Plots.resize("parityPlotDiv"), 50);
    } catch (err) {
        console.error("Parity Plot error:", err);
    }
}

function getColorRGBA(colorName, alpha = 0.2) {
    const colorMap = {
        "blue": `rgba(51, 153, 255, ${alpha})`,
        "gray": `rgba(160, 160, 160, ${alpha})`,
        "lightblue": `rgba(102, 204, 255, ${alpha})`,
        "black": `rgba(0, 0, 0, ${alpha})`,
        "red": `rgba(255, 50, 50, ${alpha})`,
        "purple": `rgba(153, 51, 255, ${alpha})`
    };
    return colorMap[colorName.toLowerCase()] || `rgba(51, 153, 255, ${alpha})`;
}

// Render GP Plot (2D Curve or 3D Surface via Matplotlib)
async function renderGPPlot() {
    const gpPlotDiv = document.getElementById("gpPlotDiv");
    if (!gpPlotDiv) return;

    const isStd = document.getElementById("chkPlotStandard").checked;
    const minInputs = document.querySelectorAll(".gp-min");
    const maxInputs = document.querySelectorAll(".gp-max");

    const varRanges = [];
    minInputs.forEach((minInp, idx) => {
        const maxInp = maxInputs[idx];
        varRanges.push({
            min: parseFloat(minInp.value) || 0,
            max: parseFloat(maxInp.value) || 10
        });
    });

    const payload = {
        standard_plot: isStd,
        n_points: parseInt(document.getElementById("selPlotPoints").value) || 1000,
        var_ranges: varRanges,
        cmap: document.getElementById("selPlotCmap").value,
        model_color: document.getElementById("selModelColor").value,
        ic_color: document.getElementById("selICColor").value
    };

    try {
        const res = await apiFetch("/api/plot-graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorText = await res.text();
            let msg = errorText;
            try {
                const jsonErr = JSON.parse(errorText);
                msg = jsonErr.detail || errorText;
            } catch (e) {}
            throw new Error(msg);
        }

        const data = await res.json();

        if (!data.train_done || !data.image) {
            gpPlotDiv.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#888; text-align:center; padding:20px;">
                    <div style="font-size:13px; font-weight:bold; margin-bottom:6px; color:#aaa;">No Model Trained</div>
                    <div style="font-size:11px;">Please click <strong>"Train Model"</strong> on the main window first.</div>
                </div>
            `;
            return;
        }

        gpPlotDiv.innerHTML = `<img src="${data.image}" style="width:100%; height:100%; object-fit:contain; display:block; margin:auto;" />`;
    } catch (err) {
        console.error("GP Plot error:", err);
        gpPlotDiv.innerHTML = `<div style="color:#ff6666; padding:15px; font-size:12px; text-align:center;">Error rendering plot:<br>${err.message}</div>`;
    }
}

// Predict Y
async function runPredictY() {
    const inputs = document.querySelectorAll(".pred-x-val");
    const xVals = Array.from(inputs).map(i => parseFloat(i.value) || 0.0);
    const confLevel = parseFloat(document.getElementById("txtCIPercent").value) || 95.0;

    try {
        const res = await apiFetch("/api/predict-y", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x_values: xVals, confidence_level: confLevel })
        });
        if (!res.ok) {
            const errorText = await res.text();
            let msg = errorText;
            try {
                const jsonErr = JSON.parse(errorText);
                msg = jsonErr.detail || errorText;
            } catch (e) {}
            throw new Error(msg);
        }
        const data = await res.json();

        document.getElementById("lblPredYVal").textContent = data.pred_y;
        document.getElementById("lblCIVal").textContent = `[${data.ci_lower} , ${data.ci_upper}]`;
    } catch (err) {
        alert("Prediction Error: " + err.message);
    }
}

// AL and BO Plot & Search
async function renderAFPlot() {
    const afPlotDiv = document.getElementById("afPlotDiv");
    if (!afPlotDiv) return;

    const isStd = document.getElementById("chkALBOStandard").checked;
    const isAvail = document.getElementById("chkALBOAvailable").checked;

    const minInputs = document.querySelectorAll(".albo-min");
    const maxInputs = document.querySelectorAll(".albo-max");

    const xRanges = [];
    minInputs.forEach((minInp, idx) => {
        const maxInp = maxInputs[idx];
        xRanges.push({
            min: parseFloat(minInp.value) || 0,
            max: parseFloat(maxInp.value) || 10
        });
    });

    const payload = {
        af_type: document.getElementById("selAFType").value,
        standard_plot: isStd,
        import_available: isAvail,
        n_points: parseInt(document.getElementById("selALBONPoints").value) || 1000,
        x_ranges: xRanges,
        cmap: document.getElementById("selALBOCmap").value,
        model_color: document.getElementById("selALBOModelColor").value,
        af_color: document.getElementById("selALBOAFColor").value
    };

    try {
        const res = await apiFetch("/api/albo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errorText = await res.text();
            let msg = errorText;
            try {
                const jsonErr = JSON.parse(errorText);
                msg = jsonErr.detail || errorText;
            } catch (e) {}
            throw new Error(msg);
        }

        const data = await res.json();

        if (data.image) {
            afPlotDiv.innerHTML = `<img src="${data.image}" style="width:100%; height:100%; object-fit:contain; display:block; margin:auto;" />`;
        } else {
            afPlotDiv.innerHTML = `
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; color:#888; text-align:center; padding:20px;">
                    <div style="font-size:13px; font-weight:bold; margin-bottom:6px; color:#aaa;">No Model Trained</div>
                    <div style="font-size:11px;">Please click <strong>"Train Model"</strong> on the main window first.</div>
                </div>
            `;
        }

        // Update Point to Measure in results
        if (data.next_point) {
            const resultSpans = document.querySelectorAll(".albo-result-val");
            data.next_point.forEach((val, idx) => {
                if (resultSpans[idx]) {
                    resultSpans[idx].textContent = (typeof val === 'number') ? val.toFixed(4) : val;
                }
            });
        }
    } catch (err) {
        console.error("AF Plot error:", err);
        afPlotDiv.innerHTML = `<div style="color:#ff6666; padding:15px; font-size:12px; text-align:center;">Error rendering AF plot:<br>${err.message}</div>`;
    }
}

async function searchALBOPoint() {
    const isAvail = document.getElementById("chkALBOAvailable").checked;
    if (isAvail && currentSession.available_data_columns.length === 0) {
        document.getElementById("availableDataFileInput").click();
        return;
    }
    await renderAFPlot();
}

// Config & Global Reset
function configReset() {
    document.getElementById("selKernel").disabled = false;
    document.getElementById("selNormLabel").disabled = false;
    document.getElementById("selNormFeat").disabled = false;
    document.getElementById("chkLikelihood").disabled = false;
    document.getElementById("chkWhiteKernel").disabled = false;
    document.getElementById("chkSplit").disabled = false;
    document.getElementById("btnTrainModel").disabled = false;
    toggleSplitInput();
}

let activeFileHandle = null;

function updateActiveFileLabel(filename) {
    const lbl = document.getElementById("activeFileLabel");
    if (lbl) {
        if (filename) {
            lbl.textContent = `File: ${filename}`;
            lbl.style.color = "#0056b3";
        } else {
            lbl.textContent = "File: (Unsaved Session)";
            lbl.style.color = "#666";
        }
    }
}

async function writeToActiveFileHandle(blob, defaultFilename) {
    if (activeFileHandle && typeof activeFileHandle.createWritable === "function") {
        try {
            const writable = await activeFileHandle.createWritable();
            await writable.write(blob);
            await writable.close();
            updateActiveFileLabel(activeFileHandle.name);
            return true;
        } catch (err) {
            console.warn("FileSystemFileHandle write failed or permission denied, falling back to download:", err);
        }
    }

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = defaultFilename || "gp_session.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    updateActiveFileLabel(defaultFilename);
    return false;
}

async function globalReset() {
    try {
        await apiFetch("/api/global-reset", { method: "POST" });
        activeFileHandle = null;
        currentSession.train_done = false;
        currentSession.type_data = "Manual";
        currentSession.n_dimensions = 2;
        currentSession.axes_titles = ["X", "Y"];
        currentSession.var_bounds = [{ name: "X", min: 0, max: 10 }];
        currentSession.csv_columns = [];

        document.querySelectorAll(".points-table input").forEach(input => {
            input.disabled = false;
        });

        document.getElementById("lblDimensions").textContent = "---";
        document.getElementById("lblTrainPoints").textContent = "---";
        document.getElementById("lblTestPoints").textContent = "---";
        document.getElementById("lblHyperparams").textContent = "---";

        configReset();
        updateAllVariableInputs();
        updateActiveFileLabel(null);
        await updateParityPlot();
        alert("Global Reset complete.");
    } catch (err) {
        alert("Reset failed: " + err.message);
    }
}

// Load Session (Native File Picker)
async function loadSession() {
    if (window.showOpenFilePicker) {
        try {
            const [handle] = await window.showOpenFilePicker({
                types: [{
                    description: "JSON Files (*.json)",
                    accept: { "application/json": [".json"] }
                }]
            });
            activeFileHandle = handle;
            const file = await handle.getFile();
            await processLoadedFile(file);
            return;
        } catch (err) {
            if (err.name === "AbortError") return;
            console.warn("showOpenFilePicker error, falling back to input:", err);
        }
    }

    document.getElementById("sessionFileInput").click();
}

async function handleLoadSessionInput(event) {
    const file = event.target.files[0];
    if (!file) return;
    activeFileHandle = null;
    await processLoadedFile(file);
    event.target.value = "";
}

async function processLoadedFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await apiFetch("/api/load-session", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to load session file");

        const sData = data.session_data;
        document.getElementById("selKernel").value = sData.var_kernel || "RBF";
        document.getElementById("selNormLabel").value = sData.var_norm_label || "None";
        document.getElementById("selNormFeat").value = sData.var_norm_feat || "None";
        document.getElementById("chkLikelihood").checked = sData.trainlikelihood || false;
        document.getElementById("chkWhiteKernel").checked = sData.white_kernel || false;
        document.getElementById("chkSplit").checked = sData.Train_Test_Split || false;
        document.getElementById("txtTestSplit").value = sData.Split_Percentage || 20;
        toggleSplitInput();

        updateModelDetails({
            n_dimensions: sData.ncol_data,
            n_train: data.n_train,
            n_test: data.n_test,
            num_params: data.num_params,
            axes_titles: sData.Axes_titles,
            var_bounds: data.var_bounds
        });

        await updateParityPlot();

        if (sData.train_done) {
            const setDisabled = (id, disabled) => {
                const el = document.getElementById(id);
                if (el) el.disabled = disabled;
            };
            setDisabled("btnTrainModel", true);
            setDisabled("selKernel", true);
            setDisabled("selNormLabel", true);
            setDisabled("selNormFeat", true);
            setDisabled("chkLikelihood", true);
            setDisabled("chkWhiteKernel", true);
            setDisabled("chkSplit", true);
            setDisabled("txtTestSplit", true);
            document.querySelectorAll(".manual-x, .manual-y").forEach(inp => inp.disabled = true);
        } else {
            configReset();
        }

        updateAllVariableInputs();
        const fname = activeFileHandle ? activeFileHandle.name : file.name;
        updateActiveFileLabel(fname);
        alert(`Session '${fname}' loaded successfully!`);
    } catch (err) {
        alert("Error loading session: " + err.message);
    }
}

// Save Session As
async function saveAsSession() {
    const suggestedName = activeFileHandle ? activeFileHandle.name : "gp_session.json";

    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: suggestedName,
                types: [{
                    description: "JSON Files (*.json)",
                    accept: { "application/json": [".json"] }
                }]
            });
            activeFileHandle = handle;
            const res = await apiFetch(`/api/save-as?filename=${encodeURIComponent(handle.name)}`, { method: "POST" });
            if (!res.ok) throw new Error("Failed to save session as JSON file.");
            const blob = await res.blob();
            const inPlace = await writeToActiveFileHandle(blob, handle.name);
            if (inPlace) {
                alert(`Session saved as '${handle.name}'!`);
            }
            return;
        } catch (err) {
            if (err.name === "AbortError") return;
            console.warn("showSaveFilePicker error, falling back to modal:", err);
        }
    }

    const input = document.getElementById("txtSaveAsFilename");
    if (input && !input.value) {
        input.value = suggestedName;
    }
    openModal("modalSaveAs");
}

async function confirmSaveAs() {
    let filename = document.getElementById("txtSaveAsFilename").value.trim();
    if (!filename) filename = "gp_session.json";
    if (!filename.endsWith(".json")) filename += ".json";

    closeModal("modalSaveAs");

    try {
        const res = await apiFetch(`/api/save-as?filename=${encodeURIComponent(filename)}`, { method: "POST" });
        if (!res.ok) throw new Error("Failed to save session as JSON file.");
        const blob = await res.blob();
        await writeToActiveFileHandle(blob, filename);
        alert(`Session saved as '${filename}'!`);
    } catch (err) {
        alert("Error saving session: " + err.message);
    }
}

// Save Session (In-Place Overwrite)
async function saveSession() {
    if (!activeFileHandle) {
        await saveAsSession();
        return;
    }

    try {
        const res = await apiFetch("/api/save", { method: "POST" });
        const contentType = res.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
            const data = await res.json();
            if (data.status === "no_save_path") {
                await saveAsSession();
                return;
            }
        }
        if (!res.ok) throw new Error("Failed to save session.");

        const blob = await res.blob();
        const inPlace = await writeToActiveFileHandle(blob, activeFileHandle.name);
        if (inPlace) {
            alert(`Session saved directly to '${activeFileHandle.name}'!`);
        }
    } catch (err) {
        alert("Error saving session: " + err.message);
    }
}

// Responsive resize on window resize
window.addEventListener("resize", () => {
    ["parityPlotDiv", "gpPlotDiv", "afPlotDiv"].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.data) {
            Plotly.Plots.resize(el);
        }
    });
});

// Initial Setup
window.addEventListener("DOMContentLoaded", async () => {
    document.querySelectorAll('.modal-overlay').forEach(modalEl => {
        makeWindowDraggable(modalEl);
    });
    updateAllVariableInputs();

    if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(() => {
            ["parityPlotDiv", "gpPlotDiv", "afPlotDiv"].forEach(id => {
                const el = document.getElementById(id);
                if (el && el.data) {
                    Plotly.Plots.resize(el);
                }
            });
        });
        document.querySelectorAll(".plot-container").forEach(c => ro.observe(c));
    }

    try {
        const res = await apiFetch("/api/model-info");
        const info = await res.json();
        updateActiveFileLabel(info.save_path);
    } catch (e) {}
    await updateParityPlot();
    setTimeout(() => {
        const el = document.getElementById("parityPlotDiv");
        if (el) Plotly.Plots.resize(el);
    }, 100);
});
