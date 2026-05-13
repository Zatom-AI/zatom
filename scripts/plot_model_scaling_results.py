"""Script to plot model scaling results including training loss, crystal validity, and molecule
validity."""

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
    "Zatom-1": {"params": 80, "label": "Zatom-1 (80M)", "color": "#009E73", "marker": "o"},
    "Zatom-1-L": {"params": 160, "label": "Zatom-1-L (160M)", "color": "#E69F00", "marker": "s"},
    "Zatom-1-XL": {"params": 300, "label": "Zatom-1-XL (300M)", "color": "#56B4E9", "marker": "D"},
}

# Define plot configurations for each row
plot_configs = [
    {
        "title": "Train loss",
        "filename": "model_scaling_train_loss.pdf",
        "inset_bounds": [0.11, 0.17, 0.35, 0.35],
        "corr_text_xy": [0.95, 0.95],
        "corr_text_ha": "right",
        "legend_loc": "upper right",
        "df": df_loss,
        "y_label_left": "Train loss ↓",
        "y_max": 3.0,
    },
    {
        "title": "Crystal validity",
        "filename": "model_scaling_crystal_validity.pdf",
        "inset_bounds": [0.63, 0.16, 0.36, 0.36],
        "corr_text_xy": [0.05, 0.95],
        "corr_text_ha": "left",
        "legend_loc": "lower left",
        "df": df_crystal,
        "y_label_left": "Crystal validity rate (%) ↑",
    },
    {
        "title": "Molecule validity",
        "filename": "model_scaling_molecule_validity.pdf",
        "inset_bounds": [0.63, 0.16, 0.36, 0.36],
        "corr_text_xy": [0.05, 0.95],
        "corr_text_ha": "left",
        "legend_loc": "lower left",
        "df": df_molecule,
        "y_label_left": "Molecule validity rate (%) ↑",
    },
]

plt.style.use("default")  # Use a standard style

# Get data at epoch 2000 for the correlation plots
epoch_2000_data = {}
for config in plot_configs:
    # Find the row closest to epoch 2000
    row_2000 = config["df"].iloc[(config["df"]["epoch"] - 2000).abs().argsort()[:1]]
    epoch_2000_data[config["title"]] = {
        model_name: row_2000[model_name].iloc[0] for model_name in models
    }


def add_model_size_inset(config, ax):
    ax_inset = ax.inset_axes(config["inset_bounds"])
    ax_inset.set_in_layout(False)

    x_vals = np.log10([props["params"] for props in models.values()])
    y_vals = [epoch_2000_data[config["title"]][model_name] for model_name in models]
    sizes = [props["params"] * 0.65 for props in models.values()]
    colors = [props["color"] for props in models.values()]
    markers = [props["marker"] for props in models.values()]

    for x, y, size, color, marker in zip(x_vals, y_vals, sizes, colors, markers):
        ax_inset.scatter(
            x,
            y,
            s=size,
            c=color,
            marker=marker,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.5,
        )

    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    ax_inset.plot(x_vals, slope * np.array(x_vals) + intercept, color="darkgrey", zorder=0)

    pearson_val, _ = pearsonr(x_vals, y_vals)
    spearman_val, _ = spearmanr(x_vals, y_vals)
    ax_inset.text(
        *config["corr_text_xy"],
        f"Pearson={pearson_val:.2f}\nSpearman={spearman_val:.2f}",
        transform=ax_inset.transAxes,
        fontsize=7,
        horizontalalignment=config["corr_text_ha"],
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, linewidth=0.5),
    )

    ax_inset.set_title("Ep. 2000 vs. size", fontsize=7, pad=2)
    ax_inset.set_xlabel("log10(params in M)", fontsize=7, labelpad=1)
    ax_inset.tick_params(axis="both", labelsize=7, length=2, pad=1)
    ax_inset.margins(x=0.2, y=0.25)


def plot_metric(config, ax, *, show_title=False):
    # --- MAIN PLOT: Metric vs. Epoch ---
    for model_name, props in models.items():
        ax.plot(
            config["df"]["epoch"],
            config["df"][model_name],
            label=props["label"],
            color=props["color"],
            marker=props["marker"],
            markersize=4,
            linestyle="-",
        )

    # Apply y-axis upper limit if provided in config.
    if config.get("y_max", None) is not None:
        ymin, _ = ax.get_ylim()
        ax.set_ylim(ymin, config["y_max"])

    # ax.axvline(x=2000, color="grey", linestyle="--", alpha=0.7)
    ax.set_xscale("symlog", linthresh=250)
    ax.set_xlim(0, 2100)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(config["y_label_left"])
    if show_title:
        ax.set_title(config["title"])
    ax.legend(loc=config["legend_loc"], fontsize=8)
    # ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    add_model_size_inset(config, ax)


output_dir = os.path.dirname(__file__)

fig, axes = plt.subplots(3, 1, figsize=(6, 9))
for config, ax in zip(plot_configs, axes):
    plot_metric(config, ax, show_title=True)

fig.subplots_adjust(left=0.14, right=0.97, bottom=0.07, top=0.95, hspace=0.55)
fig.savefig(
    os.path.join(output_dir, "model_scaling_results.pdf"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.close(fig)

for config in plot_configs:
    subplot_fig, ax = plt.subplots(figsize=(5.3, 3.3))
    plot_metric(config, ax)
    subplot_fig.subplots_adjust(left=0.17, right=0.97, bottom=0.17, top=0.96)
    subplot_fig.savefig(
        os.path.join(output_dir, config["filename"]),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(subplot_fig)
