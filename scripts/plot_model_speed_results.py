import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# --- 1. Master Configuration for All Datasets ---

# NOTE: ADiT and competitor data points are estimated from Figure 2 and Table 3 of the ADiT paper.
x_steps = np.array([10, 25, 50, 100, 250, 500, 750, 1000])

eps = 2.0  # TODO: Remove after adding real data.
dataset_config = {
    "MP20": {
        "y_label_object": "crystals",
        "y_lim": (-15, 360),
        "inset_y_lim": (-1, 21),
        "competitor_name": "FlowMM (12M)",
        "data": {
            "zatom": np.array([0.5, 0.75, 1.25, 2.25, 7, 14, 21, 28]) + eps,
            "zatom_l": np.array([1.5, 2.75, 5, 9.5, 25, 50, 75, 100]) + eps,
            "zatom_xl": np.array([4, 9, 17, 33, 85, 170, 260, 345]) + eps,
            "adit_s": np.array([0.5, 0.75, 1.25, 2.25, 7, 14, 21, 28]),
            "adit_b": np.array([1.5, 2.75, 5, 9.5, 25, 50, 75, 100]),
            "adit_l": np.array([4, 9, 17, 33, 85, 170, 260, 345]),
            "competitor": np.array([2, 4, 8, 18, 42, 75, 110, 145]),
        },
    },
    "QM9": {
        "y_label_object": "molecules",
        "y_lim": (-20, 480),
        "inset_y_lim": (-2, 22),
        "competitor_name": "GeoLDM (5M)",
        "data": {
            "zatom": np.array([1, 2, 3, 5, 11, 22, 32, 42]) + eps,
            "zatom_l": np.array([2, 4, 7, 14, 32, 60, 90, 120]) + eps,
            "zatom_xl": np.array([5, 12, 25, 45, 115, 225, 340, 450]) + eps,
            "adit_s": np.array([1, 2, 3, 5, 11, 22, 32, 42]),
            "adit_b": np.array([2, 4, 7, 14, 32, 60, 90, 120]),
            "adit_l": np.array([5, 12, 25, 45, 115, 225, 340, 450]),
            "competitor": np.array([2.5, 4.67, 7.75, 15, 34, 64, 98, 132]),
        },
    },
    "GEOM": {
        "y_label_object": "molecules",
        "y_lim": (-30, 620),
        "inset_y_lim": (-5, 45),
        "competitor_name": "SemlaFlow (46M)",
        "data": {
            "zatom": np.array([0.5, 1.75, 3.5, 7, 20, 40, 60, 80]) + eps,
            "zatom_l": np.array([1.75, 4.75, 9.5, 18, 45, 87.5, 130, 170]) + eps,
            "zatom_xl": np.array([7, 15, 30, 60, 150, 300, 450, 600]) + eps,
            "adit_s": np.array([0.5, 1.75, 3.5, 7, 20, 40, 60, 80]),
            "adit_b": np.array([1.75, 4.75, 9.5, 18, 45, 87.5, 130, 170]),
            "adit_l": np.array([7, 15, 30, 60, 150, 300, 450, 600]),
            "competitor": np.array([2, 5.5, 10.75, 20.75, 51, 100, 150, 200]),
        },
    },
}

# --- 2. Select the Dataset to Plot ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="MP20", choices=["MP20", "QM9", "GEOM"])
args = parser.parse_args()
# ------------------------------------

# Load the configuration for the selected dataset
config = dataset_config[args.dataset]
data = config["data"]

# --- 3. Setup the Plot ---
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

colors = {
    "zatom": "#525252",
    "zatom_l": "#969696",
    "zatom_xl": "#cccccc",
    "adit_s": "#1f497d",
    "adit_b": "#4f81bd",
    "adit_l": "#8cb4d9",
    "competitor": "tomato",
}

# --- 4. Plot Main Data Series using Selected Config ---
ax.plot(x_steps, data["zatom"], marker="o", color=colors["zatom"], label="Zatom (80M)")
ax.plot(
    x_steps,
    data["zatom_l"],
    marker="o",
    color=colors["zatom_l"],
    label="Zatom-L (160M)",
)
ax.plot(
    x_steps,
    data["zatom_xl"],
    marker="o",
    color=colors["zatom_xl"],
    label="Zatom-XL (300M)",
)
ax.plot(x_steps, data["adit_s"], marker="o", color=colors["adit_s"], label="ADiT-S (80M)")
ax.plot(
    x_steps,
    data["adit_b"],
    marker="o",
    color=colors["adit_b"],
    label="ADiT-B (180M)",
)
ax.plot(x_steps, data["adit_l"], marker="o", color=colors["adit_l"], label="ADiT-L (500M)")
ax.plot(
    x_steps,
    data["competitor"],
    marker="x",
    color=colors["competitor"],
    label=config["competitor_name"],
)

# --- 5. Customize Main Plot Appearance ---
y_label = f"Time to sample 10K {config['y_label_object']} (mins)"
ax.set_xlabel("Number of integration steps", fontsize=14)
ax.set_ylabel(y_label, fontsize=14)

# Set axis limits from config
ax.set_ylim(*config["y_lim"])
ax.set_xlim(0, 1050)

ax.set_xticks([10, 100, 250, 500, 750, 1000])
ax.tick_params(axis="both", which="major", labelsize=12)

# --- 6. Create and Configure the Inset Plot ---
axins = ax.inset_axes([0.08, 0.53, 0.35, 0.35])

# Plot data on the inset axes
axins.plot(x_steps, data["zatom"], marker="o", color=colors["zatom"])
axins.plot(x_steps, data["zatom_l"], marker="o", color=colors["zatom_l"])
axins.plot(x_steps, data["zatom_xl"], marker="o", color=colors["zatom_xl"])
axins.plot(x_steps, data["adit_s"], marker="o", color=colors["adit_s"])
axins.plot(x_steps, data["adit_b"], marker="o", color=colors["adit_b"])
axins.plot(x_steps, data["adit_l"], marker="o", color=colors["adit_l"])
axins.plot(x_steps, data["competitor"], marker="x", color=colors["competitor"])

# Set the view for the inset from config
x1, x2 = 5, 105
y1, y2 = config["inset_y_lim"]
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

axins.set_xticks([10, 25, 50, 100])
axins.tick_params(axis="both", which="major", labelsize=10)

# --- 7. Draw Connection Lines for the Inset ---
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# --- 8. Add Legend and Finalize ---
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=12)
fig.subplots_adjust(right=0.75)

plt.savefig(
    os.path.join(os.path.dirname(__file__), f"{args.dataset}_model_speed_results.pdf"),
    bbox_inches="tight",
    dpi=300,
)
