"""Generate paper figures for The Remembrance thesis Chapter 5."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


# =====================================================
# FIGURE 5.1: Campaign Trajectory
# =====================================================
runs = [1, 2, 6, 7, 8, 9]
run_labels = [
    "1\nBCE\nbaseline",
    "2\nBCE+ls\n3-layer+LN",
    "6\nBPR",
    "7\nBPR\n+type-aware",
    "8\nBPR\n+self-adv",
    "9\nBPR+self-adv\n+RotatE",
]
auc_values = [0.9397, 0.9646, 0.9688, 0.9662, 0.9786, 0.9759]
mrr_training_time = [0.8134, 0.8361, 0.8860, 0.8873, 0.9119, 0.9095]

mrr_run8_seeds = [0.9642, 0.9611, 0.9498, 0.9613, 0.9591, 0.9566,
                  0.9558, 0.9516, 0.9576, 0.9498, 0.9621, 0.9629]
mrr_run8_mean = float(np.mean(mrr_run8_seeds))
mrr_run8_std = float(np.std(mrr_run8_seeds, ddof=1))

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(runs, auc_values, "o-", label="AUC-ROC", color="#1f4e79", linewidth=2, markersize=9)
ax.plot(runs, mrr_training_time, "s-", label="MRR (training-time, single seed)",
        color="#c5a028", linewidth=2, markersize=9)

ax.errorbar([8.0], [mrr_run8_mean], yerr=[mrr_run8_std], fmt="D",
            color="#2d6a4f", markersize=12, capsize=8, capthick=2, linewidth=2,
            label="MRR (Run 8, 12-seed mean +/- std)")

ax.axhline(y=0.95, color="#7a1a1a", linestyle="--", linewidth=1.5, alpha=0.7,
           label="Paper target (0.95)")

ax.annotate("All 4 KPIs\nPASS at Run 8\n(multi-seed)",
            xy=(8.0, mrr_run8_mean), xytext=(7.4, 0.875),
            fontsize=10, color="#2d6a4f", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#2d6a4f", lw=1.5))

ax.set_xlabel("Run #")
ax.set_ylabel("Score")
ax.set_title("Tuning Campaign Trajectory: AUC-ROC and MRR across 9 Runs", weight="bold")
ax.set_xticks(runs)
ax.set_xticklabels(run_labels, fontsize=9)
ax.set_ylim(0.79, 1.00)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_1_campaign_trajectory.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig5_1_campaign_trajectory.png")


# =====================================================
# FIGURE 5.2: Multi-seed MRR Distribution (Run 8)
# =====================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(mrr_run8_seeds, bins=8, color="#2d6a4f", alpha=0.85,
         edgecolor="white", linewidth=1.5)
ax1.axvline(x=0.95, color="#7a1a1a", linestyle="--", linewidth=2,
            label="Paper target (0.95)")
ax1.axvline(x=mrr_run8_mean, color="#1f4e79", linestyle="-", linewidth=2.5,
            label=f"Mean = {mrr_run8_mean:.4f}")
ax1.fill_betweenx([0, 5], mrr_run8_mean - mrr_run8_std,
                  mrr_run8_mean + mrr_run8_std,
                  color="#1f4e79", alpha=0.15,
                  label=f"+/-1 std ({mrr_run8_std:.4f})")
ax1.set_xlabel("MRR (uniform eval)")
ax1.set_ylabel("Count of seeds")
ax1.set_title("Run 8 MRR Distribution (n=12 seeds)", weight="bold")
ax1.set_ylim(0, 4)
ax1.legend(loc="upper left", fontsize=9)

seeds = [0, 1, 2, 5, 7, 11, 13, 23, 31, 42, 99, 100]
sorted_pairs = sorted(zip(seeds, mrr_run8_seeds), key=lambda x: x[1])
seeds_sorted = [s for s, _ in sorted_pairs]
mrr_sorted = [m for _, m in sorted_pairs]
colors = ["#7a1a1a" if m < 0.95 else "#2d6a4f" for m in mrr_sorted]
ax2.scatter(range(len(seeds_sorted)), mrr_sorted, c=colors, s=130, alpha=0.85,
            edgecolors="white", linewidth=1.5)
ax2.axhline(y=0.95, color="#7a1a1a", linestyle="--", linewidth=1.5,
            label="Paper target (0.95)")
ax2.axhline(y=mrr_run8_mean, color="#1f4e79", linestyle="-", linewidth=2,
            label=f"Mean = {mrr_run8_mean:.4f}")
ax2.set_xticks(range(len(seeds_sorted)))
ax2.set_xticklabels([f"s={s}" for s in seeds_sorted], rotation=45, ha="right",
                    fontsize=9)
ax2.set_ylabel("MRR (uniform eval)")
ax2.set_title("Per-seed MRR: 10 of 12 >= 0.95", weight="bold")
ax2.set_ylim(0.94, 0.97)
ax2.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_2_mrr_distribution.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig5_2_mrr_distribution.png")


# =====================================================
# FIGURE 5.3: Score Distribution by Loss Function and Decoder
# =====================================================
buckets = ["<0.50", "0.50-0.85", "0.85-0.95", "0.95-0.99", ">=0.99"]

bce_counts = [4514, 1865, 40, 0, 0]
bpr_counts = [19, 59, 181, 700, 5460]
bpr_adv_counts = [18, 159, 531, 1773, 3938]
rotate_counts = [6419, 0, 0, 0, 0]

x = np.arange(len(buckets))
width = 0.20

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, bce_counts, width, label="Run 5 (BCE + label smoothing)",
       color="#a0a0a0")
ax.bar(x - 0.5*width, bpr_counts, width, label="Run 6 (BPR)", color="#1f4e79")
ax.bar(x + 0.5*width, bpr_adv_counts, width,
       label="Run 8 (BPR + self-adv) [recommended]", color="#2d6a4f")
ax.bar(x + 1.5*width, rotate_counts, width, label="Run 9 (RotatE)",
       color="#c5a028")

ax.set_xlabel("Plausibility score bucket")
ax.set_ylabel("Edge count (out of 6,419)")
ax.set_title("Score Distribution by Loss Function and Decoder\n"
             "(shows why threshold calibration is loss-dependent)",
             weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(buckets)
ax.legend(loc="upper left", fontsize=10)

ax.axvspan(2.5, 4.5, alpha=0.1, color="#7a1a1a")
ax.text(3.5, 5800,
        "Canonical paper threshold\nregion (>= 0.95)",
        ha="center", fontsize=10, color="#7a1a1a", weight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_3_score_distribution.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig5_3_score_distribution.png")


# =====================================================
# FIGURE 5.4: GNN Uplift over Prompt-Only
# =====================================================
runs_uplift = ["Run 1\n(BCE baseline)", "Run 6\n(BPR)", "Run 8\n(BPR+self-adv)"]
fullstack_g = [0.839, 0.987, 0.9884]
promptonly_g = [0.763, 0.7462, 0.6826]
fullstack_f = [0.787, 0.979, 0.9714]
promptonly_f = [0.625, 0.3057, 0.3195]

x = np.arange(len(runs_uplift))
width = 0.32

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.bar(x - width/2, fullstack_g, width,
        label="Full-Stack (Validate-then-Generate)", color="#2d6a4f")
ax1.bar(x + width/2, promptonly_g, width,
        label="Prompt-Only (Standard RAG)", color="#a0a0a0")
ax1.set_xticks(x)
ax1.set_xticklabels(runs_uplift, fontsize=9)
ax1.set_ylabel("Grounding score")
ax1.set_title("Grounding Uplift: H1 Confirmation", weight="bold")
ax1.set_ylim(0, 1.10)
ax1.axhline(y=0.98, color="#7a1a1a", linestyle="--", alpha=0.6,
            label="Paper target (0.98)")
ax1.legend(loc="lower right", fontsize=9)
for i, (fs, po) in enumerate(zip(fullstack_g, promptonly_g)):
    delta = fs - po
    pct = delta / po * 100
    ax1.text(i, max(fs, po) + 0.02, f"+{delta:.3f}\n({pct:.0f}%)",
             ha="center", fontsize=9, color="#2d6a4f", weight="bold")

ax2.bar(x - width/2, fullstack_f, width,
        label="Full-Stack (Validate-then-Generate)", color="#2d6a4f")
ax2.bar(x + width/2, promptonly_f, width,
        label="Prompt-Only (Standard RAG)", color="#a0a0a0")
ax2.set_xticks(x)
ax2.set_xticklabels(runs_uplift, fontsize=9)
ax2.set_ylabel("Faithfulness score")
ax2.set_title("Faithfulness Uplift: H1 Confirmation", weight="bold")
ax2.set_ylim(0, 1.10)
ax2.legend(loc="lower right", fontsize=9)
for i, (fs, po) in enumerate(zip(fullstack_f, promptonly_f)):
    delta = fs - po
    pct = delta / po * 100
    ax2.text(i, max(fs, po) + 0.02, f"+{delta:.3f}\n({pct:.0f}%)",
             ha="center", fontsize=9, color="#2d6a4f", weight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_4_gnn_uplift.png"),
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved fig5_4_gnn_uplift.png")

print("\nAll 4 figures generated successfully.")
