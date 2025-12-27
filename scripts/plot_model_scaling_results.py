import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# --- 1. Generate Dummy Data & CSV Files ---
# This section creates placeholder CSV files with data similar to the plots.
# If you already have these files, you can skip this part.

# Common epoch range
epochs = np.arange(0, 2001, 250)
df = pd.DataFrame({"epoch": epochs})

# a) Training Loss Data
# Starts high and decays, with larger models having slightly lower loss.
if not os.path.exists("training_losses.csv"):
    loss_s = 1.08 + 1.8 * np.exp(-epochs / 250) + np.random.normal(0, 0.01, len(epochs))
    loss_b = 1.04 + 1.8 * np.exp(-epochs / 250) + np.random.normal(0, 0.01, len(epochs))
    loss_l = 1.01 + 1.8 * np.exp(-epochs / 250) + np.random.normal(0, 0.01, len(epochs))
    loss_df = df.copy()
    loss_df["Zatom"] = loss_s
    loss_df["Zatom-L"] = loss_b
    loss_df["Zatom-XL"] = loss_l
    loss_df.to_csv("training_losses.csv", index=False)

    print("Created placeholder training_losses.csv")

# b) Crystal Validity Data
# Starts low and saturates near 1.0, with larger models performing better.
if not os.path.exists("crystal_validity.csv"):
    crystal_s = 0.92 - 0.9 * np.exp(-epochs / 400) + np.random.normal(0, 0.005, len(epochs))
    crystal_b = 0.93 - 0.9 * np.exp(-epochs / 350) + np.random.normal(0, 0.005, len(epochs))
    crystal_l = 0.94 - 0.9 * np.exp(-epochs / 300) + np.random.normal(0, 0.005, len(epochs))
    crystal_df = df.copy()
    crystal_df["Zatom"] = np.clip(crystal_s, 0, 1)
    crystal_df["Zatom-L"] = np.clip(crystal_b, 0, 1)
    crystal_df["Zatom-XL"] = np.clip(crystal_l, 0, 1)
    crystal_df.to_csv("crystal_validity.csv", index=False)

    print("Created placeholder crystal_validity.csv")

# c) Molecule Validity Data
# Similar to crystal, but with a more pronounced performance gap between models.
if not os.path.exists("molecule_validity.csv"):
    mol_s = 0.90 - 1.5 * np.exp(-epochs / 600) + np.random.normal(0, 0.01, len(epochs))
    mol_b = 0.95 - 1.2 * np.exp(-epochs / 450) + np.random.normal(0, 0.01, len(epochs))
    mol_l = 0.96 - 1.1 * np.exp(-epochs / 400) + np.random.normal(0, 0.01, len(epochs))
    mol_df = df.copy()
    mol_df["Zatom"] = np.clip(mol_s, 0, 1)
    mol_df["Zatom-L"] = np.clip(mol_b, 0, 1)
    mol_df["Zatom-XL"] = np.clip(mol_l, 0, 1)
    mol_df.to_csv("molecule_validity.csv", index=False)

    print("Created placeholder molecule_validity.csv")

# --- 2. Load Data from CSV Files ---
df_loss = pd.read_csv("training_losses.csv")
df_crystal = pd.read_csv("crystal_validity.csv")
df_molecule = pd.read_csv("molecule_validity.csv")

# --- 3. Plotting ---

# Define model properties for consistency
models = {
    "Zatom": {"params": 80, "label": "Zatom (80M)", "color": "#1f77b4"},
    "Zatom-L": {"params": 160, "label": "Zatom-L (160M)", "color": "#ff7f0e"},
    "Zatom-XL": {"params": 300, "label": "Zatom-XL (300M)", "color": "#2ca02c"},
}

# Define plot configurations for each row
plot_configs = [
    {
        "title": "Train loss",
        "df": df_loss,
        "y_label_left": "Train loss",
        "y_label_right": "Ep. 2000: Train loss",
    },
    {
        "title": "Crystal validity",
        "df": df_crystal,
        "y_label_left": "Crystal validity rate (%)",
        "y_label_right": "Ep. 2000: Crystal validity rate (%)",
    },
    {
        "title": "Molecule validity",
        "df": df_molecule,
        "y_label_left": "Molecule validity rate (%)",
        "y_label_right": "Ep. 2000: Molecule validity rate (%)",
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
            marker=".",
            markersize=6,
            linestyle="-",
        )

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

    # Scatter plot with scaled bubble sizes
    ax_right.scatter(
        x_vals, y_vals, s=sizes, c=colors, alpha=0.9, edgecolors="black", linewidth=0.5
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

    ax_right.set_xlabel("Log(Number of parameters in M)")
    ax_right.set_ylabel(config["y_label_right"])
    ax_right.yaxis.set_label_position("right")
    ax_right.yaxis.tick_right()
    ax_right.grid(True, which="both", linestyle="--", linewidth=0.5)

# Final adjustments and saving the figure
plt.tight_layout(pad=2.0)
plt.savefig("model_scaling_results.pdf", dpi=300)
