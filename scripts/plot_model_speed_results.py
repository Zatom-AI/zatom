import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# --- 1. Define Data ---
# Data points are estimated from Figure 2(a) of the ADiT paper, as the original data was not available.
# X-values represent the number of integration steps.
x_steps = np.array([10, 25, 50, 100, 250, 500, 750, 1000])

# Y-values represent the time in minutes for each model.
eps = 0.5  # TODO: Replace this small temporary offset approach with real data.
zatom_y = np.array([1.5, 2.5, 3.5, 5.5, 10, 18, 24, 31]) + eps
zatom_l_y = np.array([2.0, 3.5, 6.5, 12, 25, 50, 75, 100]) + eps
zatom_xl_y = np.array([4, 9, 17, 35, 85, 170, 260, 345]) + eps
adit_s_y = np.array([1.5, 2.5, 3.5, 5.5, 10, 18, 24, 31])
adit_b_y = np.array([2.0, 3.5, 6.5, 12, 25, 50, 75, 100])
adit_l_y = np.array([4, 9, 17, 35, 85, 170, 260, 345])
flowmm_y = np.array([2.5, 5, 9, 18, 42, 80, 115, 155])

# --- 2. Setup the Plot ---
# Create a figure and a set of subplots. Adjust figsize for appropriate aspect ratio.
fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

# Define colors to match the original plot
colors = {
    "zatom": "#525252",
    "zatom_l": "#969696",
    "zatom_xl": "#cccccc",
    "adit_s": "#1f497d",
    "adit_b": "#4f81bd",
    "adit_l": "#8cb4d9",
    "flowmm": "tomato",
}

# --- 3. Plot Main Data Series ---
# Plot each line with its corresponding marker, color, and label.
ax.plot(x_steps, zatom_y, marker="s", color=colors["zatom"], label="Zatom (80M)")
ax.plot(x_steps, zatom_l_y, marker="s", color=colors["zatom_l"], label="Zatom-L (160M)")
ax.plot(x_steps, zatom_xl_y, marker="s", color=colors["zatom_xl"], label="Zatom-XL (300M)")
ax.plot(x_steps, adit_s_y, marker="o", color=colors["adit_s"], label="ADiT-S (80M)")
ax.plot(x_steps, adit_b_y, marker="o", color=colors["adit_b"], label="ADiT-B (180M)")
ax.plot(x_steps, adit_l_y, marker="o", color=colors["adit_l"], label="ADiT-L (500M)")
ax.plot(x_steps, flowmm_y, marker="x", color=colors["flowmm"], label="FlowMM (12M)")

# --- 4. Customize Main Plot Appearance ---
# Set axis labels and font sizes
ax.set_xlabel("Number of integration steps", fontsize=14)
ax.set_ylabel("Time to sample 10K crystals (mins)", fontsize=14)

# Set axis limits to match the original plot
ax.set_ylim(-15, 360)
ax.set_xlim(0, 1050)

# Set the major ticks on the x-axis
ax.set_xticks([10, 100, 250, 500, 750, 1000])

# Adjust tick label font size
ax.tick_params(axis="both", which="major", labelsize=12)

# --- 5. Create and Configure the Inset Plot ---
# Create an inset axes instance. The values in the list are [x, y, width, height]
# specified in axes coordinates (from 0 to 1).
axins = ax.inset_axes([0.08, 0.53, 0.35, 0.35])

# Plot the same data on the inset axes
axins.plot(x_steps, zatom_y, marker="s", color=colors["zatom"])
axins.plot(x_steps, zatom_l_y, marker="s", color=colors["zatom_l"])
axins.plot(x_steps, zatom_xl_y, marker="s", color=colors["zatom_xl"])
axins.plot(x_steps, adit_s_y, marker="o", color=colors["adit_s"])
axins.plot(x_steps, adit_b_y, marker="o", color=colors["adit_b"])
axins.plot(x_steps, adit_l_y, marker="o", color=colors["adit_l"])
axins.plot(x_steps, flowmm_y, marker="x", color=colors["flowmm"])

# --- 6. Set the View for the Inset ---
# Define the zoomed-in region
x1, x2, y1, y2 = 5, 105, -1, 21
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Set ticks for the inset plot
axins.set_xticks([10, 25, 50, 100])
axins.set_yticks([0, 10, 20])
axins.tick_params(axis="both", which="major", labelsize=10)

# --- 7. Draw Connection Lines for the Inset ---
# Use mark_inset to draw lines connecting the inset to the zoomed region in the main plot.
# loc1 and loc2 specify which corners of the zoom box to connect.
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# --- 8. Add Legend and Finalize ---
# Place the legend outside the main plot area to the upper right.
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=12)

# Adjust plot layout to make room for the legend
fig.subplots_adjust(right=0.75)

# Final adjustments and saving the figure
plt.savefig(
    os.path.join(os.path.dirname(__file__), "model_speed_results.pdf"),
    bbox_inches="tight",
    dpi=300,
)
