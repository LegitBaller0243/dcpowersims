import os
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.abs(y_true - y_pred).mean()
    return rmse, mae


def save_plots(y_true, y_pred, outdir="plots"):
    """
    Presentation-Friendly Plots:
        - Predictions vs True (Test)
        - Absolute Error Over Time
        - Error Distribution Histogram
        - Cumulative Error (helps show drift)
    """

    os.makedirs(outdir, exist_ok=True)

    # -----------------------------------------------------------
    # PREP
    # -----------------------------------------------------------
    y_true_vals = y_true.values
    y_pred_vals = y_pred.values
    abs_error = np.abs(y_true_vals - y_pred_vals)
    cum_error = np.cumsum(y_true_vals - y_pred_vals)  # For drift visualization

    # -----------------------------------------------------------
    # 1. Test Predictions vs True
    # -----------------------------------------------------------
    plt.figure(figsize=(16, 6))
    plt.plot(y_true.index, y_true_vals, label="Actual Power Usage", linewidth=2)
    plt.plot(y_true.index, y_pred_vals, label="Model Prediction", alpha=0.8)
    plt.title("Model Predictions vs Actual Usage (Test Set)")
    plt.ylabel("Power (kW)")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/1_predictions_vs_true.png")
    plt.close()

    # -----------------------------------------------------------
    # 2. Absolute Error Over Time
    # -----------------------------------------------------------
    plt.figure(figsize=(16, 5))
    plt.plot(y_true.index, abs_error, color="darkred")
    plt.title("Model Absolute Error Over Time")
    plt.ylabel("Absolute Error (kW)")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{outdir}/2_absolute_error_over_time.png")
    plt.close()

    # -----------------------------------------------------------
    # 3. Error Distribution Histogram
    # -----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.hist(abs_error, bins=40, color="steelblue")
    plt.title("Distribution of Model Errors")
    plt.xlabel("Absolute Error (kW)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{outdir}/3_error_distribution.png")
    plt.close()

    # -----------------------------------------------------------
    # 4. Cumulative Error (Great for showing drift)
    # -----------------------------------------------------------
    plt.figure(figsize=(16, 5))
    plt.plot(y_true.index, cum_error, color="purple")
    plt.axhline(0, linestyle="--", color="black")
    plt.title("Cumulative Prediction Error (Drift Indicator)")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Error")
    plt.tight_layout()
    plt.savefig(f"{outdir}/4_cumulative_error.png")
    plt.close()

    print(f"[plotting.py] Finished generating presentation-friendly plots in '{outdir}/'.")
