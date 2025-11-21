const API_URL = 'http://127.0.0.1:8000';

const fileInput = document.getElementById('file-upload');
const fileNameDisplay = document.getElementById('file-name');
const btnSimulate = document.getElementById('btn-simulate');
const treatmentSelect = document.getElementById('treatment-select');
const outcomeSelect = document.getElementById('outcome-select');
const confoundersList = document.getElementById('confounders-list');
const btnAnalyze = document.getElementById('btn-analyze');
const graphContainer = document.getElementById('graph-container');
const resultsContainer = document.getElementById('results-container');
const ateValue = document.getElementById('ate-value');
const ciValue = document.getElementById('ci-value');
const refuteValue = document.getElementById('refute-value');
const sigBadge = document.getElementById('significance-badge');
const upliftTreatment = document.getElementById('uplift-treatment');
const upliftControl = document.getElementById('uplift-control');
const upliftAbsolute = document.getElementById('uplift-absolute');
const upliftRelative = document.getElementById('uplift-relative');
const upliftTreatmentSize = document.getElementById('uplift-treatment-size');
const upliftControlSize = document.getElementById('uplift-control-size');

// Event Listeners
fileInput.addEventListener('change', handleFileUpload);
btnSimulate.addEventListener('click', handleSimulation);
btnAnalyze.addEventListener('click', runAnalysis);

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    fileNameDisplay.textContent = file.name;
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
        const response = await fetch(`${API_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            populateControls(data.columns);
            alert(`Loaded ${data.rows} rows successfully.`);
        } else {
            alert('Error uploading file: ' + data.detail);
        }
    } catch (error) {
        console.error(error);
        alert('Network error uploading file.');
    } finally {
        setLoading(false);
    }
}

async function handleSimulation() {
    setLoading(true);
    try {
        const response = await fetch(`${API_URL}/simulate`, { method: 'POST' });
        const data = await response.json();
        if (response.ok) {
            fileNameDisplay.textContent = "Simulated Data";
            populateControls(data.columns);
            alert(`Generated ${data.rows} rows of synthetic data.`);
        } else {
            alert('Error simulating data: ' + data.detail);
        }
    } catch (error) {
        console.error(error);
        alert('Network error simulating data.');
    } finally {
        setLoading(false);
    }
}

function populateControls(columns) {
    // Clear existing
    treatmentSelect.innerHTML = '<option value="">Select Treatment</option>';
    outcomeSelect.innerHTML = '<option value="">Select Outcome</option>';
    confoundersList.innerHTML = '';

    columns.forEach(col => {
        // Treatment & Outcome Options
        const tOption = document.createElement('option');
        tOption.value = col;
        tOption.textContent = col;
        treatmentSelect.appendChild(tOption);

        const oOption = document.createElement('option');
        oOption.value = col;
        oOption.textContent = col;
        outcomeSelect.appendChild(oOption.cloneNode(true));

        // Confounder Checkbox
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        div.innerHTML = `
            <input type="checkbox" id="conf-${col}" value="${col}">
            <label for="conf-${col}">${col}</label>
        `;
        confoundersList.appendChild(div);
    });

    treatmentSelect.disabled = false;
    outcomeSelect.disabled = false;
    btnAnalyze.disabled = false;
}

async function runAnalysis() {
    const treatment = treatmentSelect.value;
    const outcome = outcomeSelect.value;
    const confounders = Array.from(confoundersList.querySelectorAll('input:checked')).map(cb => cb.value);

    if (!treatment || !outcome) {
        alert('Please select both Treatment and Outcome.');
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ treatment, outcome, confounders })
        });
        const data = await response.json();

        if (response.ok) {
            displayResults(data);
        } else {
            alert('Error running analysis: ' + data.detail);
        }
    } catch (error) {
        console.error(error);
        alert('Network error running analysis.');
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    // Display Graph
    graphContainer.innerHTML = `<img src="data:image/png;base64,${data.graph_image}" alt="Causal Graph" />`;

    // Display Metrics
    const res = data.result;
    ateValue.textContent = (res.estimate_value ?? 0).toFixed(4);

    // Confidence Interval
    if (res.confidence_intervals && res.confidence_intervals.length === 2 && res.confidence_intervals.every(v => v !== null && v !== undefined)) {
        const [low, high] = res.confidence_intervals;
        ciValue.textContent = `[${low.toFixed(4)}, ${high.toFixed(4)}]`;
    } else {
        ciValue.textContent = "N/A";
    }

    // Refutation
    if (res.refutation_result !== null && res.refutation_result !== undefined) {
        refuteValue.textContent = res.refutation_result.toFixed(4);
    } else {
        refuteValue.textContent = "N/A";
    }

    // Significance
    if (res.p_value !== null) {
        sigBadge.style.display = 'inline-block';
        if (res.p_value < 0.05) {
            sigBadge.className = 'badge success';
            sigBadge.textContent = `Statistically Significant (p=${res.p_value.toFixed(3)})`;
        } else {
            sigBadge.className = 'badge danger';
            sigBadge.textContent = `Not Significant (p=${res.p_value.toFixed(3)})`;
        }
    } else {
        sigBadge.style.display = 'none';
    }

    updateUplift(res.uplift_summary);
}

function updateUplift(summary) {
    if (!summary) {
        setUpliftText('--', '--', '--', '--', '--', '--');
        return;
    }

    const treatmentMean = formatNumber(summary.treatment_mean);
    const controlMean = formatNumber(summary.control_mean);
    const absolute = formatNumber(summary.absolute_uplift);
    const relative = formatNumber(summary.relative_uplift_pct, 2, '%');
    const treatmentCount = summary.treatment_count ? `${summary.treatment_count} records` : '--';
    const controlCount = summary.control_count ? `${summary.control_count} records` : '--';

    setUpliftText(treatmentMean, controlMean, absolute, relative, treatmentCount, controlCount);
}

function setUpliftText(treatment, control, absolute, relative, treatSize, controlSize) {
    upliftTreatment.textContent = treatment;
    upliftControl.textContent = control;
    upliftAbsolute.textContent = absolute;
    upliftRelative.textContent = relative;
    upliftTreatmentSize.textContent = treatSize;
    upliftControlSize.textContent = controlSize;
}

function formatNumber(value, decimals = 2, suffix = '') {
    if (typeof value === 'number' && !Number.isNaN(value)) {
        return `${value.toFixed(decimals)}${suffix}`;
    }
    return 'N/A';
}

function setLoading(isLoading) {
    if (isLoading) {
        document.body.style.cursor = 'wait';
        btnAnalyze.textContent = 'Processing...';
        btnAnalyze.disabled = true;
    } else {
        document.body.style.cursor = 'default';
        btnAnalyze.textContent = 'Run Causal Analysis';
        btnAnalyze.disabled = false;
    }
}
