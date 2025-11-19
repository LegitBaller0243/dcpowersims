"""
GPU-Accelerated ARIMA/SARIMA Pipeline using RAPIDS cuML

Requires:
    conda install -c rapidsai -c conda-forge cuml=24.12 python=3.10 cudatoolkit=12

Assumes:
    - "data.csv" contains fully cleaned dataset
    - Columns: timestamp, P_it, and exogenous regressors
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from cuml.tsa.arima import ARIMA
import cudf

import matplotlib
matplotlib.use('Agg')  # required for SLURM / headless clusters
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------------
# LOAD DF
# ------------------------------------------------------------------------------

def load_df(csv_path="data.csv"):
    """Load cleaned CSV and return pandas DataFrame."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df = df.sort_values("timestamp")
    return df


# ------------------------------------------------------------------------------
# FIND TRAIN/TEST SPLIT (LARGEST GAP)
# ------------------------------------------------------------------------------

def find_split_time(df, gap_threshold=200):
    """Find timestamp where the largest gap occurs (> threshold minutes)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["delta"] = df["timestamp"].diff().dt.total_seconds().div(60)

    gap_idx = df.index[df["delta"] > gap_threshold][0]
    split_time = df.loc[gap_idx, "timestamp"]

    return split_time


# ------------------------------------------------------------------------------
# TARGET/FEATURE SPLIT
# ------------------------------------------------------------------------------

def create_targets(df):
    """Return X (exogenous), y (target)."""
    y = df["P_it"]
    X = df.drop(columns=["P_it", "timestamp"])
    return X, y


# ------------------------------------------------------------------------------
# PREPARE DATA FOR cuML ARIMA
# ------------------------------------------------------------------------------

def prepare_data(csv_path="data.csv"):
    df = load_df(csv_path)

    split_time = find_split_time(df)

    train   = df[df["timestamp"] < split_time]
    testall = df[df["timestamp"] >= split_time]

    X_train, y_train = create_targets(train)
    X_test_all, y_test_all = create_targets(testall)

    X_train = X_train.loc[:, X_train.nunique() > 1]


    # Calibration split (first half of test period)
    n = len(X_test_all)
    mid = n // 2
    X_cal, y_cal = X_test_all.iloc[:mid], y_test_all.iloc[:mid]
    X_test, y_test = X_test_all.iloc[mid:], y_test_all.iloc[mid:]


    X_cal = X_cal[X_train.columns]
    X_test = X_test[X_train.columns]

    # Scale (cuML ARIMA supports exogenous regressors via sklearn API)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_cal_s   = scaler.transform(X_cal)
    X_test_s  = scaler.transform(X_test)

    return {
        "df": df,
        "split_time": split_time,
        "X_train": X_train,
        "X_cal": X_cal,
        "X_test": X_test,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
        "scaler": scaler,
        "X_train_s": X_train_s,
        "X_cal_s": X_cal_s,
        "X_test_s": X_test_s,
    }


# ------------------------------------------------------------------------------
# GPU SARIMA FITTING (cuML)
# ------------------------------------------------------------------------------

def fit_cuml_arima(prepped,
                   order=(1,0,1),
                   seasonal_order=(0,0,1,144)):

    # Convert to cuDF
    y_train_cu = cudf.Series(prepped["y_train"].values)
    X_train_cu = cudf.DataFrame(prepped["X_train_s"])

    # FIXED: Initialize ARIMA without passing y, then fit with both y and X
    model = ARIMA(
        y_train_cu,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=False,
        exog=X_train_cu
    )

    # Fit with both endog (y) and exogenous (X) variables
    model.fit()

    return model


# ------------------------------------------------------------------------------
# GPU PREDICTION
# ------------------------------------------------------------------------------

def predict_test(prepped, model):
    """
    Predict on the final test segment using cuML ARIMA.
    """
    X_test_cu = cudf.DataFrame(prepped["X_test_s"])
    n_forecast = prepped["y_test"].shape[0]
    
    # Use forecast with exogenous variables
    y_pred = model.forecast(n_forecast, exog=X_test_cu)

    return y_pred.to_pandas()

# ------------------------------------------------------------------------------
# PLOTTING RESULTS
# ------------------------------------------------------------------------------

def save_plots(prepped, model, sig_cols, y_pred_test, outdir="plots"):
    import os
    os.makedirs(outdir, exist_ok=True)

    # Extract true values
    y_train = prepped["y_train"]
    y_test = prepped["y_test"]

    # -----------------------------
    # 1. TRAIN FIT
    # -----------------------------
    try:
        train_fitted = model.predict_in_sample().to_pandas()

        plt.figure(figsize=(14,6))
        plt.plot(y_train.index, y_train.values, label="Train True", alpha=0.8)
        plt.plot(y_train.index, train_fitted.values, label="Train Fitted", alpha=0.8)
        plt.title("Training Fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{outdir}/train_fit.png")
        plt.close()
    except Exception as e:
        print(f"Could not generate train fit plot: {e}")

    # -----------------------------
    # 2. TEST PREDICTIONS
    # -----------------------------
    plt.figure(figsize=(14,6))
    plt.plot(y_test.index, y_test.values, label="Test True", alpha=0.8)
    plt.plot(y_test.index, y_pred_test.values, label="Test Predictions", alpha=0.8)
    plt.title("Test Predictions vs Ground Truth")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/test_predictions.png")
    plt.close()

    # -----------------------------
    # 3. RESIDUALS ON TEST
    # -----------------------------
    residuals = y_test.values - y_pred_test.values

    plt.figure(figsize=(14,5))
    plt.plot(y_test.index, residuals, label="Residuals", color="red")
    plt.axhline(0, linestyle="--", color="black")
    plt.title("Residuals (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/residuals.png")
    plt.close()

    # -----------------------------
    # 4. RESIDUAL HISTOGRAM
    # -----------------------------
    plt.figure(figsize=(10,5))
    plt.hist(residuals, bins=40, alpha=0.7)
    plt.title("Residual Distribution (Test Set)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/residual_histogram.png")
    plt.close()

    print(f"Plots saved in: {outdir}/")


# ------------------------------------------------------------------------------
# ENTRYPOINT FOR BATCH JOBS
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    # Ensure GPU is used
    print("Preparing data...")
    prepped = prepare_data("data.csv")

    print("Fitting GPU ARIMA model...")
    model = fit_cuml_arima(prepped)

    print("Predicting on test set...")
    preds = predict_test(prepped, model)

    save_plots(prepped, model, None, preds)
