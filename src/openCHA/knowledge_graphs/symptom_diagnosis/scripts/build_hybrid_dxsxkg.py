"""
fig_dxsxkg_editorial_clean.py
-----------------------------
Clean editorial-style figure for DxSxKG.

Outputs:
    figures/fig_dxsxkg_editorial_clean.pdf
    figures/fig_dxsxkg_editorial_clean.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# -----------------------------
# Style
# -----------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.linewidth": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------
# Palette
# -----------------------------
BG          = "#FFFFFF"
TEXT        = "#2E2E2E"
SUBT        = "#7A7A7A"
DIVIDER     = "#D9D9D9"
ARROW       = "#8C8C8C"

DX_FILL     = "#D9D0E0"
DX_EDGE     = "#A99CB2"

SX_FILL     = "#DFE8F1"
SX_EDGE     = "#A8B9C9"

DIST_FILL   = "#EAE1CC"
DIST_EDGE   = "#B58F33"

EDGE_SOFT   = "#CCD5DD"
EDGE_HI     = "#B58F33"

# -----------------------------
# Figure size
# -----------------------------
FIG_W_MM = 178
FIG_H_MM = 72
FIG_W_IN = FIG_W_MM / 25.4
FIG_H_IN = FIG_H_MM / 25.4

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), facecolor=BG)

ax_top = fig.add_axes([0.06, 0.36, 0.88, 0.54])
ax_bot = fig.add_axes([0.08, 0.08, 0.84, 0.18])

for ax in (ax_top, ax_bot):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# -----------------------------
# Helpers
# -----------------------------
def rounded_box(ax, cx, cy, w, h, face, edge, label,
                fontsize=6.5, lw=0.7, weight="normal"):
    patch = FancyBboxPatch(
        (cx - w/2, cy - h/2),
        w, h,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        zorder=3
    )
    ax.add_patch(patch)
    ax.text(
        cx, cy, label,
        ha="center", va="center",
        fontsize=fontsize,
        color=TEXT,
        fontweight=weight,
        zorder=4
    )

def draw_bezier(ax, x0, y0, x1, y1, color, lw=0.7, alpha=0.55):
    ctrl_x = (x0 + x1) / 2
    t = np.linspace(0, 1, 200)
    cx1, cy1 = ctrl_x, y0
    cx2, cy2 = ctrl_x, y1
    bx = (1 - t) ** 3 * x0 + 3 * (1 - t) ** 2 * t * cx1 + 3 * (1 - t) * t ** 2 * cx2 + t ** 3 * x1
    by = (1 - t) ** 3 * y0 + 3 * (1 - t) ** 2 * t * cy1 + 3 * (1 - t) * t ** 2 * cy2 + t ** 3 * y1
    ax.plot(bx, by, color=color, lw=lw, alpha=alpha, solid_capstyle="round", zorder=1)

# -----------------------------
# Top panel
# -----------------------------
ax_top.text(0.00, 0.98, "(a)", ha="left", va="top",
            fontsize=8.0, fontweight="bold", color=TEXT)

ax_top.text(0.30, 0.90, "Diagnosis nodes",
            ha="center", va="center", fontsize=6.9, color=TEXT)
ax_top.text(0.70, 0.90, "Symptom nodes",
            ha="center", va="center", fontsize=6.9, color=TEXT)
ax_top.text(0.50, 0.90, "HAS_SYMPTOM",
            ha="center", va="center", fontsize=5.3, color=SUBT, style="italic")

DX_X, SX_X = 0.30, 0.70
DX_W, DX_H = 0.25, 0.12
SX_W, SX_H = 0.20, 0.12

dx_nodes = {
    "Viral sinusitis":   0.68,
    "Strep sore throat": 0.48,
    "Viral pharyngitis": 0.28,
}
sx_nodes = {
    "Fever":           0.75,
    "Cough":           0.58,
    "Sore throat":     0.41,
    "Swollen tonsils": 0.24,
}

for label, cy in dx_nodes.items():
    rounded_box(ax_top, DX_X, cy, DX_W, DX_H, DX_FILL, DX_EDGE, label)

for label, cy in sx_nodes.items():
    if label == "Swollen tonsils":
        rounded_box(ax_top, SX_X, cy, SX_W, SX_H, DIST_FILL, DIST_EDGE, label, lw=0.9)
    else:
        rounded_box(ax_top, SX_X, cy, SX_W, SX_H, SX_FILL, SX_EDGE, label)

edges = [
    ("Viral sinusitis",   "Fever"),
    ("Viral sinusitis",   "Cough"),
    ("Strep sore throat", "Fever"),
    ("Strep sore throat", "Sore throat"),
    ("Strep sore throat", "Swollen tonsils"),
    ("Viral pharyngitis", "Cough"),
    ("Viral pharyngitis", "Sore throat"),
    ("Viral pharyngitis", "Swollen tonsils"),
]

DX_RIGHT = DX_X + DX_W / 2
SX_LEFT  = SX_X - SX_W / 2

for dx_label, sx_label in edges:
    y0 = dx_nodes[dx_label]
    y1 = sx_nodes[sx_label]
    highlight = (sx_label == "Swollen tonsils")
    draw_bezier(
        ax_top,
        DX_RIGHT, y0,
        SX_LEFT, y1,
        color=EDGE_HI if highlight else EDGE_SOFT,
        lw=1.0 if highlight else 0.7,
        alpha=0.8 if highlight else 0.55
    )

# Divider
divider = plt.Line2D(
    [0.10, 0.90], [0.32, 0.32],
    transform=fig.transFigure,
    color=DIVIDER, lw=0.55, linestyle=(0, (4, 5))
)
fig.add_artist(divider)

# -----------------------------
# Bottom panel: minimal reasoning strip
# -----------------------------
ax_bot.text(0.00, 0.98, "(b)", ha="left", va="top",
            fontsize=8.0, fontweight="bold", color=TEXT)

y_main = 0.58
y_sub  = 0.30

# x positions
x_obs = 0.12
x_cand = 0.34
x_ask = 0.56
x_yes = 0.74
x_res = 0.90

# Small pills only where helpful
rounded_box(ax_bot, x_obs, y_main, 0.13, 0.16, "#F5F2EC", "#C9C1B4", "Observed", fontsize=5.8, lw=0.7, weight="bold")
rounded_box(ax_bot, x_cand, y_main, 0.13, 0.16, "#EEE8F1", "#B7A9BC", "Candidates", fontsize=5.8, lw=0.7, weight="bold")
rounded_box(ax_bot, x_ask, y_main, 0.13, 0.16, DIST_FILL, DIST_EDGE, "Ask", fontsize=5.8, lw=0.8, weight="bold")
rounded_box(ax_bot, x_res, y_main, 0.15, 0.16, "#E6EEF5", "#A9BAC9", "Resolved", fontsize=5.8, lw=0.7, weight="bold")

# Sub-labels
ax_bot.text(x_obs, y_main - 0.10, "sore throat", ha="center", va="center", fontsize=5.1, color=TEXT)
ax_bot.text(x_cand, y_main - 0.10, r"$H(q)=2$", ha="center", va="center", fontsize=5.1, color=TEXT)
ax_bot.text(x_ask, y_main - 0.10, "swollen tonsils?", ha="center", va="center", fontsize=5.1, color=TEXT)

ax_bot.text(x_yes, y_main - 0.01, "User", ha="center", va="center", fontsize=5.8, color=TEXT, fontweight="bold")
ax_bot.text(x_yes, y_main - 0.11, "yes", ha="center", va="center", fontsize=5.1, color=TEXT)

ax_bot.text(x_res, y_main - 0.10, "Strep sore throat", ha="center", va="center", fontsize=5.1, color=TEXT)

# Arrows
arrowprops = dict(arrowstyle="->", color=ARROW, lw=0.7, mutation_scale=10)

pairs = [
    (x_obs + 0.07, x_cand - 0.07),
    (x_cand + 0.07, x_ask - 0.07),
    (x_ask + 0.07, x_yes - 0.07),
    (x_yes + 0.06, x_res - 0.08),
]
for x0, x1 in pairs:
    ax_bot.annotate("", xy=(x1, y_main), xytext=(x0, y_main), arrowprops=arrowprops)

# Only two small annotations
ax_bot.text(0.45, y_sub, "Strep sore throat, Viral pharyngitis",
            ha="center", va="center", fontsize=4.9, color=SUBT)
ax_bot.text(0.90, y_sub, r"$H(q):\ 2 \rightarrow 1$",
            ha="center", va="center", fontsize=4.9, color=SUBT)

# Save
os.makedirs("figures", exist_ok=True)
pdf_path = "figures/fig_dxsxkg_editorial_clean.pdf"
png_path = "figures/fig_dxsxkg_editorial_clean.png"

fig.savefig(pdf_path, bbox_inches="tight", dpi=300, facecolor=BG)
fig.savefig(png_path, bbox_inches="tight", dpi=300, facecolor=BG)

print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")

plt.close(fig)