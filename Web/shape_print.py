import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from typing import Union, List, Optional

warnings.filterwarnings("ignore")

# =============== Utility functions ===============
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def ensure_2d_frame(X, columns):
    """Convert the input array/Series into a 2D DataFrame and align column names."""
    if isinstance(X, pd.DataFrame):
        # If it's already a DataFrame, keep only the required columns (in column-name order)
        return X[columns]
    if isinstance(X, pd.Series):
        return pd.DataFrame([X.values], columns=columns)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return pd.DataFrame(X, columns=columns)

def _load_df(maybe_path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Support both a DataFrame input and a CSV file path."""
    if isinstance(maybe_path_or_df, pd.DataFrame):
        return maybe_path_or_df.copy()
    return pd.read_csv(str(maybe_path_or_df))

# =============== Main function ===============
def explain_mods_kernel_shap(
    df_patient: Union[str, pd.DataFrame],
    df_train: Union[str, pd.DataFrame],
    scaler_file: str,
    model_file: str,
    *,
    label_col: Optional[str] = None,
    patient_row: int = 0,
    background_size: int = 100,
    nsamples: Union[str, int] = "auto"
) -> List[str]:
    """
    Use Kernel SHAP (logit link) to explain a single patient's prediction and return a list of output strings.
    
    Parameters
    ----
    df_patient : str | pd.DataFrame
        Patient data (without labels). Supports a CSV path or a DataFrame. If multiple rows exist,
        only the row specified by patient_row is explained.
    df_train : str | pd.DataFrame
        Training data (with labels). Supports a CSV path or a DataFrame.
    scaler_file : str
        Path to the scaler joblib file.
    model_file : str
        Path to the trained model joblib file (must support predict_proba).
    label_col : str | None
        Name of the label column in the training set. If None, the last column is treated as the label by default.
    patient_row : int
        Index of the patient row to explain (default: 0).
    background_size : int
        Number of background samples for KernelExplainer (taken from the first N rows of the training set; default: 100).
    nsamples : "auto" | int
        Sampling size for shap_values; default is "auto".
    
    Returns
    ----
    List[str]
        Output consistent with the original script (a list of strings).
    """
    # -------- Load data --------
    df_patient = _load_df(df_patient)
    df_train = _load_df(df_train)

    # Label column and feature columns
    if label_col is None:
        label_col = df_train.columns[-1]
    feature_cols = [c for c in df_train.columns if c != label_col]

    # Select patient row (only explain one row)
    if not (0 <= patient_row < len(df_patient)):
        raise IndexError(f"patient_row is out of range: 0 ~ {len(df_patient)-1}")
    X_patient_series = df_patient[feature_cols].iloc[patient_row]

    # Background samples (raw, unscaled features)
    BACKGROUND_SIZE = max(1, int(background_size))
    background = df_train[feature_cols].head(BACKGROUND_SIZE).copy()

    # -------- Load scaler and model --------
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)

    # Prediction function: raw features → scale → return positive-class probability
    def model_fn(X):
        X_df = ensure_2d_frame(X, feature_cols)
        X_scaled = scaler.transform(X_df.values)
        return model.predict_proba(X_scaled)[:, 1]

    # Patient predicted probability (using the same path as SHAP)
    pred_prob = float(model_fn(X_patient_series)[0])

    # -------- Build KernelExplainer and compute SHAP --------
    # Use link='logit' to ensure: expected_value (base logit) + Σphi = logit(pred_prob)
    explainer = shap.KernelExplainer(model_fn, background, link="logit")

    # Explain only this one patient row
    shap_values_raw = explainer.shap_values(X_patient_series, nsamples=nsamples)
    # For a single row, KernelExplainer returns a 1D ndarray (or a list/ndarray of length 1)
    phi = np.asarray(shap_values_raw).reshape(-1)

    # -------- Convert to text explanations --------
    def shap_to_text_prob(explainer_obj, shap_values_1d: np.ndarray, x_single_series: pd.Series):
        """
        Roughly map SHAP values in logit space to 'percentage-point probability changes'
        and output the Top 10 explanations.
        """
        base_logit = float(np.squeeze(explainer_obj.expected_value))
        base_prob = sigmoid(base_logit)

        contributions = []
        logit_running = base_logit
        for feature, value, shap_val in zip(x_single_series.index, x_single_series.values, shap_values_1d):
            prob_without = sigmoid(logit_running)
            prob_with = sigmoid(logit_running + float(shap_val))
            delta_prob_pct = (prob_with - prob_without) * 100.0
            logit_running += float(shap_val)
            contributions.append((feature, value, delta_prob_pct))

        # Sort by absolute impact and take Top 10
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)
        explanations = []
        for feature, value, delta_prob in contributions_sorted[:10]:
            if delta_prob >= 0:
                explanations.append(f"{feature} = {value}, increased risk by {abs(delta_prob):.1f}% (approx.)")
            else:
                explanations.append(f"{feature} = {value}, decreased risk by {abs(delta_prob):.1f}% (approx.)")
        return explanations, base_prob

    explanations, base_prob = shap_to_text_prob(explainer, phi, X_patient_series)

    # -------- Additivity self-check --------
    sum_logit = float(np.squeeze(explainer.expected_value)) + float(np.sum(phi))
    pred_prob_from_sum = sigmoid(sum_logit)  # Should be ≈ pred_prob (numerical error ~1e-2 is normal)

    # -------- Assemble output --------
    output = []
    output.append(f"Background baseline probability (E[logit] → probability): {base_prob:.2%}")
    output.append(f"Predicted risk for this sample: {pred_prob:.2%}")
    output.append(f"Additivity check (sigmoid(E[logit] + Σφ) ≈ predicted probability): {pred_prob_from_sum:.4f} vs {pred_prob:.4f}")
    output.append("Top 10 features ranked by impact (approx. percentage-point probability change):")
    output.extend(explanations)

    return output
