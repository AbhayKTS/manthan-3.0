import dowhy
from dowhy import CausalModel
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
import numpy as np
import math


def safe_float(val):
    """Convert numpy/pandas values to native floats when possible."""
    if isinstance(val, (np.integer, np.floating)):
        val = val.item()

    if isinstance(val, np.ndarray):
        if val.size == 1:
            val = val.item()
        else:
            return [safe_float(x) for x in val]

    try:
        f_val = float(val)
        if math.isnan(f_val) or math.isinf(f_val):
            return None
        return f_val
    except Exception:
        return None


def compute_uplift_summary(df: pd.DataFrame, treatment: str, outcome: str):
    """Compute simple uplift diagnostics (treated vs. control means)."""
    summary = None

    if treatment not in df.columns or outcome not in df.columns:
        return summary

    outcomes = pd.to_numeric(df[outcome], errors="coerce")
    treatment_series = pd.to_numeric(df[treatment], errors="coerce")

    data = pd.DataFrame({"treatment": treatment_series, "outcome": outcomes}).dropna()
    if data.empty:
        return summary

    treated = data[data["treatment"] >= 0.5]
    control = data[data["treatment"] < 0.5]
    if treated.empty or control.empty:
        return summary

    treated_mean = treated["outcome"].mean()
    control_mean = control["outcome"].mean()
    absolute_uplift = treated_mean - control_mean

    treated_var = treated["outcome"].var(ddof=1)
    control_var = control["outcome"].var(ddof=1)
    treated_n = len(treated)
    control_n = len(control)

    se = None
    if treated_n > 1 and control_n > 1:
        se_val = max(treated_var / treated_n + control_var / control_n, 0)
        if not math.isnan(se_val):
            se = math.sqrt(se_val)

    approx_ci = None
    if se is not None and se > 0:
        z = 1.96
        approx_ci = [absolute_uplift - z * se, absolute_uplift + z * se]

    relative = None
    if control_mean not in (None, 0):
        try:
            relative = (absolute_uplift / control_mean) * 100.0
        except ZeroDivisionError:
            relative = None

    summary = {
        "treatment_mean": safe_float(treated_mean),
        "control_mean": safe_float(control_mean),
        "absolute_uplift": safe_float(absolute_uplift),
        "relative_uplift_pct": safe_float(relative),
        "treatment_count": int(treated_n),
        "control_count": int(control_n),
        "standard_error": safe_float(se),
        "approximate_confidence_interval": [safe_float(x) for x in approx_ci] if approx_ci else None,
    }

    return summary

def estimate_causal_effect(df: pd.DataFrame, treatment: str, outcome: str, confounders: list):
    """
    Runs the full Causal Inference pipeline: Model -> Identify -> Estimate -> Refute.
    """
    # Sanitize inputs
    confounders = [c for c in confounders if c != treatment and c != outcome]
    uplift_summary = compute_uplift_summary(df, treatment, outcome)

    # 1. Create Causal Model
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        common_causes=confounders
    )
    
    # 2. Identify Causal Effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    # 3. Estimate Causal Effect
    # Using Linear Regression as a robust default
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True,
        confidence_intervals=True
    )
    
    # 4. Refute Estimate (Robustness Check)
    # Placebo Treatment Refuter
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )

    conf_ints = estimate.get_confidence_intervals()
    if conf_ints is not None:
        conf_ints = [safe_float(x) for x in conf_ints]

    if (not conf_ints or any(val is None for val in conf_ints)) and uplift_summary and uplift_summary.get("approximate_confidence_interval"):
        conf_ints = uplift_summary["approximate_confidence_interval"]

    return {
        "estimate_value": safe_float(estimate.value),
        "confidence_intervals": conf_ints,
        "p_value": safe_float(estimate.test_stat_significance()['p_value']) if estimate.test_stat_significance() else None,
        "refutation_result": safe_float(refutation.new_effect) if refutation is not None else None,
        "uplift_summary": uplift_summary
    }

def get_causal_graph_image(df: pd.DataFrame, treatment: str, outcome: str, confounders: list) -> str:
    """
    Generates a visual representation of the causal graph.
    Returns a base64 encoded PNG string.
    """
    # Sanitize inputs
    confounders = [c for c in confounders if c != treatment and c != outcome]

    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node(treatment, color='skyblue', style='filled')
    G.add_node(outcome, color='lightgreen', style='filled')
    for c in confounders:
        G.add_node(c, color='lightgrey', style='filled')
        G.add_edge(c, treatment)
        G.add_edge(c, outcome)
        
    G.add_edge(treatment, outcome)
    
    # Draw
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrows=True)
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64
