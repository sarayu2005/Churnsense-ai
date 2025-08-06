# causal_utils.py
import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import matplotlib.pyplot as plt
import os

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

def save_plot(fig, filename):
    """Save plot and return path"""
    path = f"plots/{filename}"
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path.replace("\\", "/")

def run_causal_analysis(df: pd.DataFrame, treatment: str, outcome: str, common_causes: list):
    """
    Run a complete causal analysis pipeline using DoWhy.
    """
    try:
        # 1. Preprocess data (similar to ML utils)
        # Ensure outcome is binary
        if df[outcome].dtype != 'int64' and df[outcome].dtype != 'int32':
             df[outcome] = (df[outcome] == df[outcome].unique()[0]).astype(int)

        # Ensure treatment is numeric (for this example, we assume it is)
        df[treatment] = pd.to_numeric(df[treatment], errors='coerce').fillna(0)

        # Drop non-numeric common causes for this simplified model
        numeric_common_causes = df[common_causes].select_dtypes(include=np.number).columns.tolist()
        
        # 2. Create a Causal Model from the data
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=numeric_common_causes
        )
        
        # 3. Identify the causal effect
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # 4. Estimate the causal effect using a suitable method
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
            test_significance=True
        )
        
        # 5. Refute the obtained estimate
        refute_results = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause"
        )

        # 6. Visualize the causal graph
        graph_path = "plots/causal_graph.png"
        model.view_model(layout="dot", file_name=graph_path.replace(".png", ""))
        
        # Format results
        causal_results = {
            "estimated_effect": estimate.value,
            "estimate_summary": str(estimate),
            "refutation_results": str(refute_results),
            "causal_graph_url": "/" + graph_path
        }
        
        return causal_results

    except Exception as e:
        # Return a dictionary with the error message
        return {"error": f"Causal analysis failed: {str(e)}"}

