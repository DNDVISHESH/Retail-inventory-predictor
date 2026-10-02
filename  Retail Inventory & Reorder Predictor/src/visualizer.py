"""Visual Analytics Dashboard Generator for Retail Inventory & Reorder Predictor."""

import os
import numpy as np
import pandas as pd

# Safe Matplotlib configuration to avoid permission warnings on macOS sandbox
from .config import BASE_DIR, REPORT_CHART_PATH
os.environ["MPLCONFIGDIR"] = os.path.join(BASE_DIR, ".mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt


def generate_visual_dashboard(
    master_df: pd.DataFrame,
    simulation_results: dict = None,
    output_path: str = REPORT_CHART_PATH,
) -> str:
    """
    Generates a 4-panel visual dashboard of inventory health and reorder predictions.

    Panels:
    1. Inventory Health Distribution (Donut Chart)
    2. ABC Pareto Analysis (Revenue Concentration Curve)
    3. Most Urgent SKUs by Stockout Runway (Days of Inventory)
    4. Simulation On-Hand Inventory Trajectory (Predictive vs Naive)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0F172A")  # Dark slate background
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    text_color = "#F8FAFC"
    grid_color = "#334155"

    for ax in axes.flat:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # -------------------------------------------------------------
    # Panel 1: Inventory Health Distribution (Donut Chart)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    status_counts = master_df["inventory_status"].value_counts()
    status_order = ["CRITICAL", "REORDER NEEDED", "OPTIMAL", "OVERSTOCKED"]
    counts = [status_counts.get(s, 0) for s in status_order]
    colors = ["#EF4444", "#F59E0B", "#10B981", "#6366F1"]

    # Filter non-zeros
    filtered_data = [(s, c, col) for s, c, col in zip(status_order, counts, colors) if c > 0]
    labels = [f"{s} ({c})" for s, c, _ in filtered_data]
    vals = [c for _, c, _ in filtered_data]
    cols = [col for _, _, col in filtered_data]

    wedges, texts, autotexts = ax1.pie(
        vals,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=cols,
        textprops=dict(color=text_color, fontsize=9, fontweight="bold"),
        wedgeprops=dict(width=0.45, edgecolor="#0F172A", linewidth=2),
    )
    for at in autotexts:
        at.set_color("#FFFFFF")
    ax1.set_title("1. Inventory Health Status Distribution", color=text_color, fontsize=12, fontweight="bold", pad=12)

    # -------------------------------------------------------------
    # Panel 2: ABC Pareto Revenue Concentration
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    sorted_df = master_df.sort_values(by="annual_projected_revenue", ascending=False).reset_index(drop=True)
    total_rev = sorted_df["annual_projected_revenue"].sum()
    cum_rev_pct = (sorted_df["annual_projected_revenue"].cumsum() / total_rev) * 100.0
    sku_pct = ((sorted_df.index + 1) / len(sorted_df)) * 100.0

    ax2.plot(sku_pct, cum_rev_pct, color="#38BDF8", linewidth=2.5, label="Cumulative Revenue %")
    ax2.plot([0, 100], [0, 100], color="#64748B", linestyle="--", linewidth=1.2, label="Equal Distribution")
    ax2.axhline(80, color="#EF4444", linestyle=":", linewidth=1.5, label="Class A Cutoff (80%)")
    ax2.axhline(95, color="#F59E0B", linestyle=":", linewidth=1.5, label="Class B Cutoff (95%)")

    ax2.set_xlabel("% of Total SKUs", color=text_color, fontsize=10)
    ax2.set_ylabel("Cumulative Revenue %", color=text_color, fontsize=10)
    ax2.set_title("2. ABC Pareto Analysis (Revenue Concentration)", color=text_color, fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, color=grid_color, linestyle="--", alpha=0.5)
    ax2.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor=text_color, fontsize=8, loc="lower right")

    # -------------------------------------------------------------
    # Panel 3: Most Urgent SKUs by Stockout Runway (Days Remaining)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    top_urgent = master_df.sort_values(by="priority_score", ascending=False).head(12)
    y_pos = np.arange(len(top_urgent))

    # Color code bars by status
    bar_colors = [
        "#EF4444" if st == "CRITICAL" else "#F59E0B" if st == "REORDER NEEDED" else "#10B981"
        for st in top_urgent["inventory_status"]
    ]

    bars = ax3.barh(y_pos, top_urgent["stockout_runway_days"], color=bar_colors, height=0.65, edgecolor="#0F172A")
    ax3.set_yticks(y_pos)
    labels_y = [f"{row['sku_id']} ({row['product_name'][:18]}...)" for _, row in top_urgent.iterrows()]
    ax3.set_yticklabels(labels_y, fontsize=8)
    ax3.invert_yaxis()  # Highest priority on top
    ax3.axvline(7.0, color="#EF4444", linestyle="--", linewidth=1.5, label="Critical Runway (7 Days)")

    ax3.set_xlabel("Days of Inventory Remaining (Runway)", color=text_color, fontsize=10)
    ax3.set_title("3. Top 12 Urgent SKUs Requiring Reorder", color=text_color, fontsize=12, fontweight="bold", pad=12)
    ax3.grid(True, axis="x", color=grid_color, linestyle="--", alpha=0.5)
    ax3.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor=text_color, fontsize=8, loc="lower right")

    # -------------------------------------------------------------
    # Panel 4: Simulation Timeline Comparison (Predictive vs Naive)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    if simulation_results and "predictive" in simulation_results and "naive" in simulation_results:
        days_range = np.arange(1, simulation_results["simulation_days"] + 1)
        pred_line = simulation_results["predictive"]["on_hand_timeline"]
        naive_line = simulation_results["naive"]["on_hand_timeline"]

        ax4.plot(days_range, pred_line, color="#10B981", linewidth=2.2, label="Predictive ROP/EOQ Policy")
        ax4.plot(days_range, naive_line, color="#F43F5E", linewidth=2.0, linestyle="--", label="Naive Periodic Policy")
        ax4.set_xlabel("Simulation Day", color=text_color, fontsize=10)
        ax4.set_ylabel("Total Units In Stock (Storewide)", color=text_color, fontsize=10)
        ax4.set_title("4. 45-Day Inventory Trajectory Simulation", color=text_color, fontsize=12, fontweight="bold", pad=12)
        ax4.grid(True, color=grid_color, linestyle="--", alpha=0.5)
        ax4.legend(facecolor="#1E293B", edgecolor="#475569", labelcolor=text_color, fontsize=8, loc="upper right")
    else:
        # Fallback: category breakdown
        cat_counts = master_df.groupby("category")["reorder_cost"].sum().sort_values(ascending=False)
        ax4.bar(cat_counts.index, cat_counts.values, color="#38BDF8")
        ax4.set_title("Reorder Investment by Category ($)", color=text_color, fontsize=12, fontweight="bold", pad=12)
        ax4.tick_params(axis="x", rotation=25)

    plt.suptitle("Retail Inventory & Reorder Optimization Dashboard (NumPy & Pandas)", color=text_color, fontsize=16, fontweight="bold", y=0.98)

    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path
