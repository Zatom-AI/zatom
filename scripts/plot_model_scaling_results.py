import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- 1. Defining Data ---
df_loss = pd.DataFrame(
    {
        "epoch": {
            0: 0,
            1: 250,
            2: 500,
            3: 750,
            4: 1000,
            5: 1250,
            6: 1500,
            7: 1750,
            8: 2000,
        },
        "Zatom-1": {
            0: 19.64182,
            1: 1.75808,
            2: 1.1903,
            3: 1.124,
            4: 1.04569,
            5: 0.99231,
            6: 0.94617,
            7: 0.93031,
            8: 0.90646,
        },
        "Zatom-1-L": {
            0: 10.66805,
            1: 1.56799,
            2: 1.24015,
            3: 1.10702,
            4: 1.00602,
            5: 0.95718,
            6: 0.92139,
            7: 0.91278,
            8: 0.88432,
        },
        "Zatom-1-XL": {
            0: 2.10083,
            1: 1.90358,
            2: 1.34222,
            3: 1.15693,
            4: 1.0264,
            5: 0.94898,
            6: 0.90873,
            7: 0.89944,
            8: 0.86318,
        },
    }
)
df_crystal = pd.DataFrame(
    {
        "epoch": {
            0: 0,
            1: 250,
            2: 500,
            3: 750,
            4: 1000,
            5: 1250,
            6: 1500,
            7: 1750,
            8: 2000,
        },
        "Zatom-1": {
            0: 0.0,
            1: 0.81709,
            2: 0.85793,
            3: 0.86321,
            4: 0.87091,
            5: 0.87986,
            6: 0.88825,
            7: 0.89036,
            8: 0.89395,
        },
        "Zatom-1-L": {
            0: 0.0,
            1: 0.82945,
            2: 0.85425,
            3: 0.87011,
            4: 0.88129,
            5: 0.88723,
            6: 0.89203,
            7: 0.89399,
            8: 0.89774,
        },
        "Zatom-1-XL": {
            0: 0.0,
            1: 0.78988,
            2: 0.84974,
            3: 0.86944,
            4: 0.88591,
            5: 0.89744,
            6: 0.90045,
            7: 0.90265,
            8: 0.90272,
        },
    }
)
df_molecule = pd.DataFrame(
    {
        "epoch": {
            0: 0,
            1: 250,
            2: 500,
            3: 750,
            4: 1000,
            5: 1250,
            6: 1500,
            7: 1750,
            8: 2000,
        },
        "Zatom-1": {
            0: 0.0,
            1: 0.61825,
            2: 0.87266,
            3: 0.90917,
            4: 0.92757,
            5: 0.93686,
            6: 0.94389,
            7: 0.94408,
            8: 0.94678,
        },
        "Zatom-1-L": {
            0: 0.0,
            1: 0.76158,
            2: 0.87094,
            3: 0.91805,
            4: 0.93684,
            5: 0.94547,
            6: 0.94673,
            7: 0.94759,
            8: 0.9476,
        },
        "Zatom-1-XL": {
            0: 0.0,
            1: 0.55776,
            2: 0.82099,
            3: 0.89122,
            4: 0.92563,
            5: 0.94033,
            6: 0.94668,
            7: 0.94999,
            8: 0.94935,
        },
    }
)

# --- 2. Plotting ---

# Define model properties for consistency
models = {
    "Zatom-1": {"params": 80, "label": "Zatom-1 (80M)", "color": "#1f77b4", "marker": "o"},
    "Zatom-1-L": {"params": 160, "label": "Zatom-1-L (160M)", "color": "#ff7f0e", "marker": "s"},
    "Zatom-1-XL": {"params": 300, "label": "Zatom-1-XL (300M)", "color": "#2ca02c", "marker": "D"},
}

# Define plot configurations for each row
plot_configs = [
    {
        "title": "Train loss",
        "df": df_loss,
        "y_label_left": "Train loss ↓",
        "y_label_right": "Ep. 2000: Train loss ↓",
        "y_max": 3.0,
    },
    {
        "title": "Crystal validity",
        "df": df_crystal,
        "y_label_left": "Crystal validity rate (%) ↑",
        "y_label_right": "Ep. 2000: Crystal validity rate (%) ↑",
    },
    {
        "title": "Molecule validity",
        "df": df_molecule,
        "y_label_left": "Molecule validity rate (%) ↑",
        "y_label_right": "Ep. 2000: Molecule validity rate (%) ↑",
    },
]

# Create the 3x2 subplot grid
fig, axes = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={"width_ratios": [2.5, 1]})
plt.style.use("default")  # Use a standard style

# Get data at epoch 2000 for the correlation plots
epoch_2000_data = {}
for config in plot_configs:
    # Find the row closest to epoch 2000
    row_2000 = config["df"].iloc[(config["df"]["epoch"] - 2000).abs().argsort()[:1]]
    epoch_2000_data[config["title"]] = {
        model_name: row_2000[model_name].iloc[0] for model_name in models
    }

# Loop through each row configuration to create the plots
for i, config in enumerate(plot_configs):
    ax_left = axes[i, 0]
    ax_right = axes[i, 1]

    # --- LEFT PLOT: Metric vs. Epoch ---
    for model_name, props in models.items():
        ax_left.plot(
            config["df"]["epoch"],
            config["df"][model_name],
            label=props["label"],
            color=props["color"],
            marker=props["marker"],
            markersize=4,
            linestyle="-",
        )
        # Apply y-axis upper limit if provided in config (plotting functions don't accept y_max)
        if config.get("y_max", None) is not None:
            ymin, _ = ax_left.get_ylim()
            ax_left.set_ylim(ymin, config["y_max"])

    ax_left.axvline(x=2000, color="grey", linestyle="--", alpha=0.7)
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel(config["y_label_left"])
    ax_left.legend()
    ax_left.grid(True, which="both", linestyle="--", linewidth=0.5)

    # --- RIGHT PLOT: Correlation at Epoch 2000 ---
    x_vals = np.log10([props["params"] for props in models.values()])
    y_vals = [epoch_2000_data[config["title"]][model_name] for model_name in models]
    sizes = [props["params"] * 3 for props in models.values()]  # Scale bubble size
    colors = [props["color"] for props in models.values()]
    markers = [props["marker"] for props in models.values()]

    # Scatter plot with scaled bubble sizes
    for x, y, size, color, marker in zip(x_vals, y_vals, sizes, colors, markers):
        ax_right.scatter(
            x, y, s=size, c=color, marker=marker, alpha=0.9, edgecolors="black", linewidth=0.5
        )

    # Calculate and plot trend line
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    ax_right.plot(x_vals, slope * np.array(x_vals) + intercept, color="darkgrey", zorder=0)

    # Calculate and display correlations
    pearson_val, _ = pearsonr(x_vals, y_vals)
    spearman_val, _ = spearmanr(x_vals, y_vals)
    corr_text = f"Pearson: {pearson_val:.2f}\nSpearman: {spearman_val:.2f}"
    ax_right.text(
        0.05,
        0.95,
        corr_text,
        transform=ax_right.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.5),
    )

    ax_right.set_xlabel("log(Number of parameters in M)")
    ax_right.set_ylabel(config["y_label_right"])
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.grid(True, which="both", linestyle="--", linewidth=0.5)

# Final adjustments and saving the figure
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(os.path.dirname(__file__), "model_scaling_results.pdf"), dpi=300)
