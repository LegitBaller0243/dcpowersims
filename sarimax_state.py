"""
SARIMAX with Fourier terms and improved differencing for data center power

Key ideas:
- Chronological train/test split (last 20% = test by default)
- Model the *differenced* target (ΔP_it) to enforce stationarity
- Difference only drifting exogenous variables (GPU temp, clocks, util, etc.)
- Keep Fourier seasonal terms undifferenced (they're deterministic)
- Use statsmodels SARIMAX on CPU
- Reconstruct level forecasts for the test period
"""

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

from plotting import save_plots  # your existing helpers


# ======================================================================
# 1. BASIC HELPERS
# ======================================================================

def load_df(csv_path: str) -> pd.DataFrame:
    """Load and sort the raw dataframe."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Chronological split: first train_frac as train, rest as test.
    """
    n = len(df)
    split_idx = int(n * train_frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


# ======================================================================
# 2. FOURIER FEATURES
# ======================================================================

def add_fourier_terms(df: pd.DataFrame,
                      period: int = 24,
                      K: int = 3,
                      prefix: str = "fourier_daily_") -> pd.DataFrame:
    """
    Add Fourier seasonal terms for a given period.

    period = seasonal period in time steps (24 for daily if data is hourly)
    K = number of harmonics
    """
    df = df.copy()
    t = np.arange(len(df))

    for k in range(1, K + 1):
        df[f"{prefix}sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"{prefix}cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return df


def add_weekly_fourier_terms(df: pd.DataFrame,
                             period: int = 24 * 7,
                             K: int = 1,
                             prefix: str = "fourier_weekly_") -> pd.DataFrame:
    """
    Add lower-frequency weekly Fourier terms (optional but often helpful).
    """
    df = df.copy()
    t = np.arange(len(df))

    for k in range(1, K + 1):
        df[f"{prefix}sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        df[f"{prefix}cos_{k}"] = np.cos(2 * np.pi * k * t / period)

    return df


# ======================================================================
# 3. DRIFT / RESIDUAL DIAGNOSTICS
# ======================================================================

def diagnose_drift(y_train: pd.Series, y_test: pd.Series, label: str = "P_it"):
    """Print basic drift diagnostics on the *level* target."""
    print("\n" + "=" * 60)
    print("DRIFT DIAGNOSTICS")
    print("=" * 60)

    train_mean = y_train.mean()
    test_mean = y_test.mean()
    train_std = y_train.std()
    test_std = y_test.std()

    print(f"\nTarget Variable ({label}):")
    print(f"  Train: mean={train_mean:.2f}, std={train_std:.2f}")
    print(f"  Test:  mean={test_mean:.2f}, std={test_std:.2f}")
    print(f"  Mean shift: {test_mean - train_mean:.2f} "
          f"({100*(test_mean-train_mean)/train_mean:.1f}%)")
    print(f"  Variance ratio: {test_std/train_std:.3f}")

    # Drift inside test set
    half = len(y_test) // 2
    if half > 0:
        first_half = y_test.iloc[:half]
        second_half = y_test.iloc[half:]
        first_mean = first_half.mean()
        second_mean = second_half.mean()
        print(f"\n  Test first half mean: {first_mean:.2f}")
        print(f"  Test second half mean: {second_mean:.2f}")
        print(f"  Drift within test: {second_mean - first_mean:.2f}")

    # Linear trend over test
    t = np.arange(len(y_test))
    if len(t) > 1:
        coef = np.polyfit(t, y_test.values, 1)
        trend = coef[0]
        print(f"  Test period linear trend: {trend:.4f} per step")
        print(f"  Total expected drift: {trend * len(y_test):.2f}")

    print("=" * 60 + "\n")


def analyze_residual_drift(y_true: pd.Series, y_pred: pd.Series, label: str = "Test"):
    """Check if residuals (in levels) still have drift or autocorrelation."""
    residuals = y_true - y_pred
    t = np.arange(len(residuals))

    coef = np.polyfit(t, residuals.values, 1)
    trend = coef[0]
    total_drift = trend * len(residuals)

    print(f"\nResidual Analysis ({label}):")
    print(f"  Mean residual: {residuals.mean():.2f}")
    print(f"  Residual std: {residuals.std():.2f}")
    print(f"  Residual trend: {trend:.4f} per step")
    print(f"  Total drift: {total_drift:.2f}")

    if len(residuals) > 1:
        r = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        print(f"  Residual lag-1 autocorr: {r:.3f}")

    return trend


# ======================================================================
# 4. DATA PREPARATION WITH DIFFERENCING
# ======================================================================

def prepare_data_with_fourier(csv_path: str, train_frac: float = 0.8):
    """
    End-to-end preparation:
    - Load data
    - Add Fourier terms
    - Difference the target and drifting exogs
    - Build model-ready matrices (y_diff, X_diff)
    - Chronological split on differenced data
    - Also return level-series for evaluation / plotting
    """
    df = load_df(csv_path)

    # Add Fourier seasonal terms (daily + weekly)
    df = add_fourier_terms(df, period=24, K=3, prefix="daily_")
    df = add_weekly_fourier_terms(df, period=24 * 7, K=1, prefix="weekly_")

    drifting_cols = [
        "GPU_Temp_Avg",
        "GPU_Clock_Avg",
        "Load Average",
        "cpus_util_cfg",
        "gpus_util_cfg",
        "memory_util_cfg",
    ]
    drifting_cols = [c for c in drifting_cols if c in df.columns]

    fourier_cols = [c for c in df.columns if c.startswith("daily_") or c.startswith("weekly_")]

    work = df[["timestamp", "P_it"] + drifting_cols + fourier_cols].copy()
    work.set_index("timestamp", inplace=True)

    y_level_full = work["P_it"].copy()

    work["y_diff"] = work["P_it"].diff()

    for col in drifting_cols:
        work[col + "_diff"] = work[col].diff()

    exog_model_cols = [col + "_diff" for col in drifting_cols] + fourier_cols

    work_model = work.dropna(subset=["y_diff"])

    y_diff_full = work_model["y_diff"]
    X_full = work_model[exog_model_cols]

    # ✅ SAFE ALIGNMENT
    common_idx = y_diff_full.index.intersection(y_level_full.index)
    y_diff_full = y_diff_full.loc[common_idx]
    X_full = X_full.loc[common_idx]
    y_level_aligned = y_level_full.loc[common_idx]

    n = len(y_diff_full)
    split_idx = int(n * train_frac)

    train_idx = y_diff_full.index[:split_idx]
    test_idx = y_diff_full.index[split_idx:]

    y_train_diff = y_diff_full.loc[train_idx]
    y_test_diff = y_diff_full.loc[test_idx]

    X_train = X_full.loc[train_idx]
    X_test = X_full.loc[test_idx]

    y_train_level = y_level_aligned.loc[train_idx]
    y_test_level = y_level_aligned.loc[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    diagnose_drift(y_train_level, y_test_level, label="P_it")

    return {
        "y_train_diff": y_train_diff,
        "y_test_diff": y_test_diff,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "y_train_level": y_train_level,
        "y_test_level": y_test_level,
        "y_diff_full": y_diff_full,
        "X_full_s": np.vstack([X_train_s, X_test_s]),
        "train_idx": train_idx,
        "test_idx": test_idx,
        "y_level_full": y_level_full,
        "exog_model_cols": exog_model_cols,
        "scaler": scaler,
    }



# ======================================================================
# 5. MODEL FIT + INVERTING DIFFERENCES BACK TO LEVELS
# ======================================================================

def fit_sarimax(prepped,
                order=(2, 0, 2),
                seasonal_order=(1, 0, 1, 24)):
    """
    Fit SARIMAX to the differenced target and exogenous variables.
    """
    y_train = prepped["y_train_diff"]
    X_train_s = prepped["X_train_s"]

    model = SARIMAX(
        endog=y_train,
        exog=X_train_s,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    results = model.fit(disp=False)
    print("\nModel fitted.")
    print(results.summary())
    return results


def invert_differences(diff_series: pd.Series, y0: float) -> pd.Series:
    """
    Given a differenced series Δy_t and an initial level y_0 (the level
    *before* the first difference), reconstruct the level series.
    """
    levels = []
    prev = y0
    for delta in diff_series.values:
        prev = prev + delta
        levels.append(prev)
    return pd.Series(levels, index=diff_series.index)


def make_level_predictions(prepped, results):
    """
    Produce level predictions for train and test segments.

    - Use fittedvalues for train (on differenced scale).
    - Use get_forecast for test (on differenced scale).
    - Then invert differences to recover levels.
    """
    y_diff_full = prepped["y_diff_full"]
    train_idx = prepped["train_idx"]
    test_idx = prepped["test_idx"]
    y_level_full = prepped["y_level_full"]

    X_test_s = prepped["X_test_s"]

    # Fitted diff values for train
    fitted_diff_train = results.fittedvalues.loc[train_idx]

    # Forecast diff for test
    forecast_res = results.get_forecast(steps=len(test_idx), exog=X_test_s)
    pred_diff_test = forecast_res.predicted_mean
    pred_diff_test.index = test_idx  # ensure matching index

    # --- Invert training differences back to levels ---

    # First training diff is at time t1; its true level is y_level_full[t1]
    # and y_diff[t1] = y_level[t1] - y_level[t0]
    first_train_t = train_idx[0]
    y_first_train_level = y_level_full.loc[first_train_t]
    first_diff_value = y_diff_full.loc[first_train_t]
    # Level *before* first diff:
    y0_train = y_first_train_level - first_diff_value

    y_train_pred_level = invert_differences(fitted_diff_train, y0_train)
    y_train_true_level = y_level_full.loc[train_idx]


    # The "previous" level for the first test diff is the last level of train
    last_train_t = train_idx[-1]
    y_last_train_level = y_level_full.loc[last_train_t]
    y0_test = y_last_train_level

    y_test_pred_level = invert_differences(pred_diff_test, y0_test)
    y_test_true_level = y_level_full.loc[test_idx]

    return (y_train_true_level, y_train_pred_level,
            y_test_true_level, y_test_pred_level)


# ======================================================================
# 6. METRICS
# ======================================================================

def compute_metrics_with_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# ======================================================================
# 7. ENTRYPOINT
# ======================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SARIMAX with Fourier Terms & Improved Differencing")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------------
    prepped = prepare_data_with_fourier("data/data.csv", train_frac=0.8)

    # ------------------------------------------------------------------
    # Fit model on differenced series
    # ------------------------------------------------------------------
    print("\nTraining SARIMAX...")
    print("  order=(2,1,2), seasonal_order=(1,1,1,24)")
    results = fit_sarimax(
        prepped,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 1, 24),
    )

    # ------------------------------------------------------------------
    # Get level predictions for train and test
    # ------------------------------------------------------------------
    (y_train_true, y_train_pred,
     y_test_true, y_test_pred) = make_level_predictions(prepped, results)

    # ------------------------------------------------------------------
    # Evaluate on test (in LEVELS, not differences)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST PERFORMANCE (LEVELS)")
    print("=" * 60)

    rmse, mae, r2 = compute_metrics_with_r2(y_test_true, y_test_pred)
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R²:   {r2:.4f}")

    analyze_residual_drift(y_test_true, y_test_pred, label="Test")

    # ------------------------------------------------------------------
    # Also show train metrics (sanity check for over/underfit)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAIN PERFORMANCE (LEVELS)")
    print("=" * 60)

    rmse_tr, mae_tr, r2_tr = compute_metrics_with_r2(y_train_true, y_train_pred)
    print(f"  RMSE: {rmse_tr:.3f}")
    print(f"  MAE:  {mae_tr:.3f}")
    print(f"  R²:   {r2_tr:.4f}")

    analyze_residual_drift(y_train_true, y_train_pred, label="Train")

    # ------------------------------------------------------------------
    # Save plots using your existing plotting utilities
    # ------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # `save_plots` was previously:
    #   save_plots(y_true_test, y_pred_test, y_train, y_train_pred, outdir)
    save_plots(
        y_true=y_test_true,
        y_pred=y_test_pred,
        outdir="plots_cpu",
    )

    print("\n" + "=" * 60)
    print("Plots saved to 'plots/' directory")
    print("=" * 60 + "\n")
