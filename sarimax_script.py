"""
GPU-Accelerated SARIMA Pipeline using RAPIDS cuML
Enhanced Version:
    - Proper differencing (non-seasonal + seasonal)
    - Stable SARIMA orders
    - Differenced exogenous regressors
    - Chronological split (not gap-based)
    - No calibration set
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from cuml.tsa.arima import ARIMA
import cudf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# 1. LOAD DF
# ==============================================================================

def load_df(csv_path="data.csv"):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ==============================================================================
# 2. TRAIN/TEST SPLIT  (CHRONOLOGICAL)
# ==============================================================================

def chronological_split(df, train_frac=0.8):
    """
    Use time-based chronological split — more stable than gap-based splitting.
    """
    n = len(df)
    split_idx = int(n * train_frac)

    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    return train, test


# ==============================================================================
# 3. DIFFERENCING (TARGET + EXOG)
# ==============================================================================

def create_xy(df, target="P_it", seasonal_period=144):
    """
    IMPORTANT:
    - We do NOT manually difference the target y.
    - ARIMA will handle differencing with order=(1,1,1), seasonal_order=(0,1,1,s)
    - We DO manually difference the exogenous regressors (1 diff + seasonal diff)
    """

    # Exogenous features
    X = df.drop(columns=["timestamp", target])

    # Difference exogenous variables
    X = X.diff().diff(seasonal_period).dropna()

    # Target y (NOT differenced)
    y = df[target].loc[X.index]

    return X, y


# ==============================================================================
# 4. PREPARE DATA
# ==============================================================================

def prepare_data(csv_path="data.csv", seasonal_period=144):
    df = load_df(csv_path)

    # DIFFERENCE ENTIRE DATASET FIRST
    X_all, y_all = create_xy(df, seasonal_period=seasonal_period)

    # SPLIT AFTER DIFFERENCING
    split_idx = int(0.8 * len(y_all))

    X_train = X_all.iloc[:split_idx]
    y_train = y_all.iloc[:split_idx]

    X_test  = X_all.iloc[split_idx:]
    y_test  = y_all.iloc[split_idx:]

    # SCALE
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_s": X_train_s,
        "X_test_s": X_test_s,
        "scaler": scaler
    }



# ==============================================================================
# 5. FIT GPU SARIMA (cuML)
# ==============================================================================

def fit_gpu_sarima(prepped,
                   order=(1,1,1),
                   seasonal_order=(0,1,1,144)):

    y_train_cu = cudf.Series(prepped["y_train"].values)
    X_train_cu = cudf.DataFrame(prepped["X_train_s"])

    model = ARIMA(
        y_train_cu,
        order=order,
        seasonal_order=seasonal_order,
        fit_intercept=True,
        exog=X_train_cu
    )

    model.fit()
    return model


# ==============================================================================
# 6. GPU FORECAST
# ==============================================================================

def predict_gpu(prepped, model):
    X_test_cu = cudf.DataFrame(prepped["X_test_s"])
    n = len(prepped["y_test"])

    y_pred = model.forecast(n, exog=X_test_cu)
    return y_pred.to_pandas()


# ==============================================================================
# 7. PLOTTING
# ==============================================================================

def save_plots(prepped, y_pred, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    y_true = prepped["y_test"]
    residuals = y_true.values - y_pred.values

    # Predictions
    plt.figure(figsize=(16,6))
    plt.plot(y_true.index, y_true.values, label="True")
    plt.plot(y_true.index, y_pred.values, label="Pred", alpha=0.7)
    plt.title("Differenced Predictions vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/test_predictions.png")
    plt.close()

    # Residuals
    plt.figure(figsize=(16,5))
    plt.plot(y_true.index, residuals, color="red")
    plt.axhline(0, ls="--", color="black")
    plt.title("Residuals")
    plt.tight_layout()
    plt.savefig(f"{outdir}/residuals.png")
    plt.close()

    print(f"Saved plots in '{outdir}/'")


# ==============================================================================
# 8. ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":

    print("Preparing data with proper differencing...")
    prepped = prepare_data("data.csv")

    print("Training stable GPU SARIMA...")
    model = fit_gpu_sarima(prepped,
                           order=(1,1,1),
                           seasonal_order=(0,1,1,144))

    print("Forecasting...")
    preds = predict_gpu(prepped, model)

    save_plots(prepped, preds)
