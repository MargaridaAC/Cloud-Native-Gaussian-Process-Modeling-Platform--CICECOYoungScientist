// State Variables
let currentSession = {
    train_done: false,
    type_data: "Manual",
    n_dimensions: 2,
    axes_titles: ["X", "Y"],
    csv_columns: [],
    uploaded_csv_preview: null
};

// Dark theme layout configuration for Plotly
const plotlyDarkLayout = {
    paper_bgcolor: "#333333",
    plot_bgcolor: "#1a1a1a",
    font: { family: "Calibri, sans-serif", color: "#ffffff", size: 12 },
    margin: { l: 45, r: 25, t: 35, b: 45 },
    xaxis: { gridcolor: "#333333", zerolinecolor: "#555555" },
    yaxis: { gridcolor: "#333333", zerolinecolor: "#555555" }
};

// Modal Control
function openModal(id) {
    document.getElementById(id).style.display = "flex";
    if (id === 'modalPlot') {
        renderGPPlot();
    } else if (id === 'modalALBO') {
        renderAFPlot();
    } else if (id === 'modalPred') {
        setupPredInputs();
    }
}

function closeModal(id) {
    document.getElementById(id).style.display = "none";
}

// Split Checkbox toggle
function toggleSplitInput() {
    const chk = document.getElementById("chkSplit");
    const txt = document.getElementById("txtTestSplit");
    txt.disabled = !chk.checked;
}

// Handle CSV Upload
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

// Train Model (Manual or CSV)
async function trainModel() {
    if (currentSession.type_data === "Import") {
        alert("Currently in CSV mode. Re-import CSV or reset to Manual.");
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

    document.getElementById("lblDimensions").textContent = data.n_dimensions;
    document.getElementById("lblTrainPoints").textContent = data.n_train;
    document.getElementById("lblTestPoints").textContent = data.n_test;
    document.getElementById("lblHyperparams").textContent = data.num_params;
}

// Parity Plot Update
async function updateParityPlot() {
    try {
        const res = await fetch("/api/parity");
        const data = await res.json();

        if (!data.train_done) {
            Plotly.newPlot("parityPlotDiv", [], {
                ...plotlyDarkLayout,
                title: "Parity plot"
            });
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
            marker: { color: "red", size: 6, symbol: "circle-open" }
        };

        if (document.getElementById("chkErrorBars").checked && data.train.y_std) {
            trainTrace.error_y = {
                type: "data",
                array: data.train.y_std,
                visible: true,
                color: "black"
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
                marker: { color: "blue", size: 6, symbol: "x" }
            };

            if (document.getElementById("chkErrorBars").checked && data.test.y_std) {
                testTrace.error_y = {
                    type: "data",
                    array: data.test.y_std,
                    visible: true,
                    color: "black"
                };
            }
            traces.push(testTrace);
        }

        // 1:1 Reference Line
        traces.push({
            x: [data.line_min, data.line_max],
            y: [data.line_min, data.line_max],
            mode: "lines",
            name: "Ideal",
            line: { color: "black", dash: "dash", width: 1 }
        });

        // Annotations for metrics
        const annotations = [];
        let yPos = 0.95;

        const m = data.metrics;
        if (document.getElementById("chkMAE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Train) = ${m.MAE_Train.toFixed(3)}`, showarrow: false, font: { color: "red" } });
            yPos -= 0.08;
        }
        if (document.getElementById("chkMAPE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Train) = ${m.MAPE_Train.toFixed(3)}`, showarrow: false, font: { color: "red" } });
            yPos -= 0.08;
        }
        if (document.getElementById("chkR2").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Train) = ${m.R2_Train.toFixed(3)}`, showarrow: false, font: { color: "red" } });
            yPos -= 0.08;
        }
        if (document.getElementById("chkRMSE").checked) {
            annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Train) = ${m.RMSE_Train.toFixed(3)}`, showarrow: false, font: { color: "red" } });
            yPos -= 0.08;
        }

        if (data.has_test) {
            if (document.getElementById("chkMAE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAE (Test) = ${m.MAE_Test.toFixed(3)}`, showarrow: false, font: { color: "blue" } });
                yPos -= 0.08;
            }
            if (document.getElementById("chkMAPE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `MAPE (Test) = ${m.MAPE_Test.toFixed(3)}`, showarrow: false, font: { color: "blue" } });
                yPos -= 0.08;
            }
            if (document.getElementById("chkR2").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `R² (Test) = ${m.R2_Test.toFixed(3)}`, showarrow: false, font: { color: "blue" } });
                yPos -= 0.08;
            }
            if (document.getElementById("chkRMSE").checked) {
                annotations.push({ x: 0.02, y: yPos, xref: "paper", yref: "paper", text: `RMSE (Test) = ${m.RMSE_Test.toFixed(3)}`, showarrow: false, font: { color: "blue" } });
                yPos -= 0.08;
            }
        }

        const layout = {
            ...plotlyDarkLayout,
            title: "Parity plot",
            xaxis: { ...plotlyDarkLayout.xaxis, title: `Exp. ${labelY}` },
            yaxis: { ...plotlyDarkLayout.yaxis, title: `Pred. ${labelY}` },
            annotations: annotations
        };

        Plotly.newPlot("parityPlotDiv", traces, layout);
    } catch (err) {
        console.error("Parity Plot error:", err);
    }
}

// Render GP Plot (2D or 3D)
async function renderGPPlot() {
    const payload = {
        standard_plot: document.getElementById("chkPlotStandard").checked,
        n_points: parseInt(document.getElementById("selPlotPoints").value) || 1000,
        var_ranges: [
            { min: parseFloat(document.getElementById("txtVar1Min").value) || 0, max: parseFloat(document.getElementById("txtVar1Max").value) || 10 },
            { min: parseFloat(document.getElementById("txtVar2Min").value) || 0, max: parseFloat(document.getElementById("txtVar2Max").value) || 10 }
        ]
    };

    try {
        const res = await fetch("/api/plot-graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (!data.train_done) {
            Plotly.newPlot("gpPlotDiv", [], { ...plotlyDarkLayout, title: "GRAPH" });
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
                line: { color: mColor }
            });

            // 95% Confidence Interval
            traces.push({
                x: data.x_plot.concat(data.x_plot.slice().reverse()),
                y: data.y_upper.concat(data.y_lower.slice().reverse()),
                fill: "toself",
                fillcolor: "rgba(0, 102, 255, 0.15)",
                line: { color: icColor, dash: "dash" },
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
                    marker: { color: "blue", size: 6 }
                });
            }

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                xaxis: { ...plotlyDarkLayout.xaxis, title: titles[0] },
                yaxis: { ...plotlyDarkLayout.yaxis, title: titles[1] }
            };

            Plotly.newPlot("gpPlotDiv", traces, layout);
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
                    marker: { color: "blue", size: 4 }
                });
            }

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                scene: {
                    xaxis: { title: titles[0], backgroundcolor: "#1a1a1a", gridcolor: "#333" },
                    yaxis: { title: titles[1], backgroundcolor: "#1a1a1a", gridcolor: "#333" },
                    zaxis: { title: titles[2] || "Y", backgroundcolor: "#1a1a1a", gridcolor: "#333" }
                }
            };

            Plotly.newPlot("gpPlotDiv", traces, layout);
        }
    } catch (err) {
        console.error("GP Plot error:", err);
    }
}

// Setup & Run Predict Y
function setupPredInputs() {
    const list = document.getElementById("predInputsList");
    list.innerHTML = "";

    const nFeat = Math.max(1, currentSession.n_dimensions - 1);
    for (let i = 0; i < nFeat; i++) {
        const input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.className = "pred-x-val";
        input.style.width = "65px";
        input.value = "0";
        list.appendChild(input);
    }
}

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

// Render AF Plot & Run AL/BO
async function renderAFPlot() {
    const payload = {
        af_type: document.getElementById("selAFType").value,
        standard_plot: document.getElementById("chkALBOStandard").checked,
        import_available: document.getElementById("chkALBOAvailable").checked,
        n_points: parseInt(document.getElementById("selALBONPoints").value) || 1000
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
                line: { color: mColor }
            });
            traces.push({
                x: data.x_plot,
                y: data.af_plot,
                mode: "lines",
                name: "AF",
                line: { color: afColor, dash: "dash" }
            });

            const layout = {
                ...plotlyDarkLayout,
                title: data.graph_title,
                xaxis: { ...plotlyDarkLayout.xaxis, title: titles[0] },
                yaxis: { ...plotlyDarkLayout.yaxis, title: "A.F." }
            };

            Plotly.newPlot("afPlotDiv", traces, layout);
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
                    xaxis: { title: titles[0], backgroundcolor: "#1a1a1a" },
                    yaxis: { title: titles[1], backgroundcolor: "#1a1a1a" },
                    zaxis: { title: "A.F.", backgroundcolor: "#1a1a1a" }
                }
            };

            Plotly.newPlot("afPlotDiv", traces, layout);
        }

        // Update Next Point Results
        const container = document.getElementById("alboLimitsContainer");
        container.innerHTML = "";
        data.next_point.forEach((val, idx) => {
            const div = document.createElement("div");
            div.className = "form-row";
            div.innerHTML = `
                <span>${titles[idx] || 'Var ' + (idx + 1)} :</span>
                <span class="albo-result" style="font-weight: bold; color: #00ffcc;">${val}</span>
            `;
            container.appendChild(div);
        });

    } catch (err) {
        console.error("AF Plot error:", err);
    }
}

async function searchALBOPoint() {
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
}

async function globalReset() {
    try {
        await fetch("/api/global-reset", { method: "POST" });
        currentSession.train_done = false;
        currentSession.type_data = "Manual";
        currentSession.n_dimensions = 2;

        document.getElementById("lblDimensions").textContent = "---";
        document.getElementById("lblTrainPoints").textContent = "---";
        document.getElementById("lblTestPoints").textContent = "---";
        document.getElementById("lblHyperparams").textContent = "---";

        configReset();
        await updateParityPlot();
        alert("Global Reset complete.");
    } catch (err) {
        alert("Reset failed: " + err.message);
    }
}

// Initial Setup
window.addEventListener("DOMContentLoaded", () => {
    updateParityPlot();
});
