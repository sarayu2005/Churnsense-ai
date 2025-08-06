import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt
import os
# Add this new function right after your imports
def preprocess_for_survival(df, time_col, churn_col):
    """Ensure time and churn columns are numeric."""
    
    # 1. Drop rows where time or churn column has missing values
    df_clean = df.dropna(subset=[time_col, churn_col]).copy()

    # 2. Convert the churn column to binary (1/0)
    #    This identifies the most common value as the "positive" case (churn=1)
    #    You can manually change 'positive_label' if needed. e.g., positive_label = 'Yes'
    if df_clean[churn_col].dtype == 'object':
        # Find the unique values and assume the first one is the "churn" event
        unique_values = df_clean[churn_col].unique()
        if len(unique_values) > 2:
            print(f"Warning: Churn column '{churn_col}' has more than two unique values: {unique_values}")

        positive_label = unique_values[0]
        print(f"Survival Analysis: Treating '{positive_label}' as the churn event (1), and others as no-event (0).")
        df_clean[churn_col] = (df_clean[churn_col] == positive_label).astype(int)

    # 3. Ensure time column is numeric
    df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
    df_clean.dropna(subset=[time_col], inplace=True) # Drop rows if time could not be converted
    
    return df_clean

os.makedirs("plots", exist_ok=True)

def save_plot(fig, filename):
    path = f"plots/{filename}"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

def run_kaplan_meier(df, time_col, churn_col):
    df_processed = preprocess_for_survival(df, time_col, churn_col) # <-- ADD THIS
    kmf = KaplanMeierFitter()
    kmf.fit(df_processed[time_col], event_observed=df_processed[churn_col]) # <-- USE df_processed
    # ... rest of function
    fig, ax = plt.subplots()
    kmf.plot_survival_function(ax=ax)
    ax.set_title("Kaplan-Meier Survival Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    return save_plot(fig, "km_curve.png")

def run_cox_model(df, time_col, churn_col):
    df_processed = preprocess_for_survival(df, time_col, churn_col) # <-- ADD THIS
    cph = CoxPHFitter(penalizer=0.1)
    # Note: Pass the original time_col and churn_col names to get_dummies `columns` parameter
    # to avoid trying to one-hot encode them.
    features = [col for col in df_processed.columns if col not in [time_col, churn_col]]
    df_encoded = pd.get_dummies(df_processed, columns=features, drop_first=True) # <-- USE df_processed & be specific
    cph.fit(df_encoded, duration_col=time_col, event_col=churn_col)
    # ... rest of function
    fig, ax = plt.subplots()
    cph.plot(ax=ax)
    ax.set_title("Cox Proportional Hazards")
    return save_plot(fig, "cox_model.png"), cph

def get_risk_scores(cph, df, time_col, churn_col):
    # This function now needs time_col and churn_col to properly preprocess
    df_processed = preprocess_for_survival(df, time_col, churn_col) # <-- ADD THIS
    features = [col for col in df_processed.columns if col not in [time_col, churn_col]]
    df_encoded = pd.get_dummies(df_processed, columns=features, drop_first=True) # <-- USE df_processed
    
    # Ensure columns match the model's expected columns
    model_cols = cph.params_.index
    df_encoded = df_encoded.reindex(columns=model_cols, fill_value=0)

    risk_scores = cph.predict_partial_hazard(df_encoded)
    return risk_scores.sort_values(ascending=False).head(10).to_dict()