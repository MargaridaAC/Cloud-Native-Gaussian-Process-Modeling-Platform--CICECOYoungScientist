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
    
    // Prevent plot shifting: trigger resize after container is visible
    setTimeout(() => {
        if (id === 'modalPlot') {
            const plotDiv = document.getElementById('gpPlotDiv');
            if (plotDiv) Plotly.Plots.resize(plotDiv);
            renderGPPlot();
        } else if (id === 'modalALBO') {
            const plotDiv = document.getElementById('afPlotDiv');
            if (plotDiv) Plotly.Plots.resize(plotDiv);
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
        const res = await fetch("/api/upload-csv", {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Failed to parse CSV file");

        currentSession.csv_columns = data.columns;
        populateCSVModal(data.columns);
        openModal("modalCSVVars");
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
        const res = await fetch("/api/upload-available-data", {
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
        const res = await fetch("/api/confirm-available-data", {
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

// Train Model (Manual)
async function trainModel() {
    if (currentSession.type_data === "Import") {
        alert("Currently in CSV mode. Re-import CSV or perform Global Reset for Manual mode.");
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
        const res = await fetch("/api/train-manual", {
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
        const res = await fetch("/api/train-csv", {
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

// Parity Plot Update - Purely Reactive
async function updateParityPlot() {
    const plotDiv = document.getElementById("parityPlotDiv");
    if (!plotDiv) return;

    try {
        const res = await fetch("/api/parity");
        const data = await res.json();

        if (!data.train_done) {
            Plotly.newPlot("parityPlotDiv", [], {
                ...plotlyDarkLayout,
                title: "Parity plot"
            }, plotlyConfig);
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
        if (document.getElementById("chkMAE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Train) = ${m.MAE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
            yPos -= 0.07;
        }
        if (document.getElementById("chkMAPE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Train) = ${m.MAPE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
            yPos -= 0.07;
        }
        if (document.getElementById("chkR2").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Train) = ${m.R2_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
            yPos -= 0.07;
        }
        if (document.getElementById("chkRMSE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Train) = ${m.RMSE_Train.toFixed(3)}`, showarrow: false, font: { color: "red", size: 10 } });
            yPos -= 0.07;
        }

        if (data.has_test) {
            if (document.getElementById("chkMAE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Test) = ${m.MAE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkMAPE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Test) = ${m.MAPE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkR2").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Test) = ${m.R2_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                yPos -= 0.07;
            }
            if (document.getElementById("chkRMSE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Test) = ${m.RMSE_Test.toFixed(3)}`, showarrow: false, font: { color: "#3399ff", size: 10 } });
                yPos -= 0.07;
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
    } catch (err) {
        console.error("Parity Plot error:", err);
    }
}

// Render GP Plot (2D Curve or 3D Surface)
async function renderGPPlot() {
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
        var_ranges: varRanges
    };

    try {
        const res = await fetch("/api/plot-graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (!data.train_done) {
            Plotly.newPlot("gpPlotDiv", [], { ...plotlyDarkLayout, title: "GRAPH" }, plotlyConfig);
            return;
        }

        const traces = [];
        const titles = data.axes_titles;
        const mColor = document.getElementById("selModelColor").value;
        const icColor = document.getElementById("selICColor").value;

        if (data.n_features === 1) {
            // 2D Curve plot
            traces.push({
                x: data.x_plot,
                y: data.y_mean,
                mode: "lines",
                name: "Y mean",
                line: { color: mColor, width: 2 }
            });

            // 95% Confidence Interval band
            traces.push({
                x: data.x_plot.concat(data.x_plot.slice().reverse()),
                y: data.y_upper.concat(data.y_lower.slice().reverse()),
                fill: "toself",
                fillcolor: "rgba(51, 153, 255, 0.2)",
                line: { color: icColor, dash: "dash", width: 1 },
                name: "Y I.C. 95%"
            });

            // Train Points
            traces.push({
                x: data.train_points.x.map(p => p[0]),
                y: data.train_points.y,
                mode: "markers",
                name: "Train",
                marker: { color: "red", size: 6 }
            });

            if (data.has_test && data.test_points) {
                traces.push({
                    x: data.test_points.x.map(p => p[0]),
                    y: data.test_points.y,
                    mode: "markers",
                    name: "Test",
                    marker: { color: "#3399ff", size: 6, symbol: "x" }
                });
            }

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                xaxis: { ...plotlyDarkLayout.xaxis, title: titles[0] },
                yaxis: { ...plotlyDarkLayout.yaxis, title: titles[1] }
            };

            Plotly.newPlot("gpPlotDiv", traces, layout, plotlyConfig);
        } else if (data.n_features === 2) {
            // 3D Surface Plot
            const meshTrace = {
                x: data.x1_plot,
                y: data.x2_plot,
                z: data.y_mean,
                type: "mesh3d",
                colorscale: document.getElementById("selPlotCmap").value.toLowerCase(),
                name: "Model Surface"
            };
            traces.push(meshTrace);

            // Train Points 3D
            traces.push({
                x: data.train_points.x.map(p => p[0]),
                y: data.train_points.x.map(p => p[1]),
                z: data.train_points.y,
                mode: "markers",
                type: "scatter3d",
                name: "Train",
                marker: { color: "red", size: 4 }
            });

            if (data.has_test && data.test_points) {
                traces.push({
                    x: data.test_points.x.map(p => p[0]),
                    y: data.test_points.x.map(p => p[1]),
                    z: data.test_points.y,
                    mode: "markers",
                    type: "scatter3d",
                    name: "Test",
                    marker: { color: "#3399ff", size: 4 }
                });
            }

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                scene: {
                    xaxis: { title: titles[0], backgroundcolor: "#000000", gridcolor: "#333" },
                    yaxis: { title: titles[1], backgroundcolor: "#000000", gridcolor: "#333" },
                    zaxis: { title: titles[2] || "Y", backgroundcolor: "#000000", gridcolor: "#333" }
                }
            };

            Plotly.newPlot("gpPlotDiv", traces, layout, plotlyConfig);
        }

        Plotly.Plots.resize("gpPlotDiv");
    } catch (err) {
        console.error("GP Plot error:", err);
    }
}

// Predict Y
async function runPredictY() {
    const inputs = document.querySelectorAll(".pred-x-val");
    const xVals = Array.from(inputs).map(i => parseFloat(i.value) || 0.0);
    const confLevel = parseFloat(document.getElementById("txtCIPercent").value) || 95.0;

    try {
        const res = await fetch("/api/predict-y", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x_values: xVals, confidence_level: confLevel })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Prediction failed");

        document.getElementById("lblPredYVal").textContent = data.pred_y;
        document.getElementById("lblCIVal").textContent = `[${data.ci_lower} , ${data.ci_upper}]`;
    } catch (err) {
        alert("Prediction Error: " + err.message);
    }
}

// AL and BO Plot & Search
async function renderAFPlot() {
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
        x_ranges: xRanges
    };

    try {
        const res = await fetch("/api/albo", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (!res.ok) throw new Error(data.detail || "AL/BO execution failed");

        const traces = [];
        const titles = data.axes_titles;
        const mColor = document.getElementById("selALBOModelColor").value;
        const afColor = document.getElementById("selALBOAFColor").value;

        if (data.n_features === 1) {
            traces.push({
                x: data.x_plot,
                y: data.y_mean,
                mode: "lines",
                name: "Y mean",
                line: { color: mColor, width: 2 }
            });
            traces.push({
                x: data.x_plot,
                y: data.af_plot,
                mode: "lines",
                name: "AF",
                line: { color: afColor, dash: "dash", width: 2 }
            });

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                xaxis: { ...plotlyDarkLayout.xaxis, title: titles[0] },
                yaxis: { ...plotlyDarkLayout.yaxis, title: "A.F." }
            };

            Plotly.newPlot("afPlotDiv", traces, layout, plotlyConfig);
        } else if (data.n_features === 2) {
            traces.push({
                x: data.x1_plot,
                y: data.x2_plot,
                z: data.af_plot,
                type: "mesh3d",
                colorscale: document.getElementById("selALBOCmap").value.toLowerCase(),
                name: "A.F. Surface"
            });

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                scene: {
                    xaxis: { title: titles[0], backgroundcolor: "#000000", gridcolor: "#333" },
                    yaxis: { title: titles[1], backgroundcolor: "#000000", gridcolor: "#333" },
                    zaxis: { title: "A.F.", backgroundcolor: "#000000", gridcolor: "#333" }
                }
            };

            Plotly.newPlot("afPlotDiv", traces, layout, plotlyConfig);
        }

        // Update Point to Measure in results
        const resultSpans = document.querySelectorAll(".albo-result-val");
        data.next_point.forEach((val, idx) => {
            if (resultSpans[idx]) {
                resultSpans[idx].textContent = val;
            }
        });

        Plotly.Plots.resize("afPlotDiv");
    } catch (err) {
        console.error("AF Plot error:", err);
    }
}

async function searchALBOPoint() {
    const isAvail = document.getElementById("chkALBOAvailable").checked;
    if (isAvail && currentSession.available_data_columns.length === 0) {
        // Prompt for available data CSV
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
    a.download = defaultFilename || "gp_session.pkl";
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(url);
    updateActiveFileLabel(defaultFilename);
    return false;
}

async function globalReset() {
    try {
        await fetch("/api/global-reset", { method: "POST" });
        activeFileHandle = null;
        currentSession.train_done = false;
        currentSession.type_data = "Manual";
        currentSession.n_dimensions = 2;
        currentSession.axes_titles = ["X", "Y"];
        currentSession.var_bounds = [{ name: "X", min: 0, max: 10 }];

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
                    description: "Pickle & JSON Files (*.pkl, *.json)",
                    accept: { "application/octet-stream": [".pkl"], "application/json": [".json"] }
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
        const res = await fetch("/api/load-session", {
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
    const suggestedName = activeFileHandle ? activeFileHandle.name : "gp_session.pkl";

    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName: suggestedName,
                types: [{
                    description: "Pickle Files (*.pkl)",
                    accept: { "application/octet-stream": [".pkl"] }
                }]
            });
            activeFileHandle = handle;
            const res = await fetch(`/api/save-as?filename=${encodeURIComponent(handle.name)}`, { method: "POST" });
            if (!res.ok) throw new Error("Failed to save session as pickle file.");
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
    if (!filename) filename = "gp_session.pkl";
    if (!filename.endsWith(".pkl")) filename += ".pkl";

    closeModal("modalSaveAs");

    try {
        const res = await fetch(`/api/save-as?filename=${encodeURIComponent(filename)}`, { method: "POST" });
        if (!res.ok) throw new Error("Failed to save session as pickle file.");
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
        const res = await fetch("/api/save", { method: "POST" });
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
    try {
        const res = await fetch("/api/model-info");
        const info = await res.json();
        updateActiveFileLabel(info.save_path);
    } catch (e) {}
    await updateParityPlot();
});
