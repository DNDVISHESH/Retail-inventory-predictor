"""Reporting and Purchase Order Exporting Module."""

import os
from typing import Optional
import numpy as np
import pandas as pd
from .config import PO_EXPORT_PATH


def print_executive_summary(master_df: pd.DataFrame) -> None:
    """Prints a styled executive KPI summary to the console."""
    total_skus = len(master_df)
    status_counts = master_df["inventory_status"].value_counts()
    critical_count = status_counts.get("CRITICAL", 0)
    reorder_count = status_counts.get("REORDER NEEDED", 0)
    optimal_count = status_counts.get("OPTIMAL", 0)
    overstock_count = status_counts.get("OVERSTOCKED", 0)

    total_stock_units = master_df["current_stock"].sum()
    total_stock_value = (master_df["current_stock"] * master_df["unit_cost"]).sum()
    total_reorder_units = master_df["recommended_order_qty"].sum()
    total_reorder_spend = master_df["reorder_cost"].sum()
    annual_rev_est = master_df["annual_projected_revenue"].sum()

    print("\n" + "=" * 80)
    print(" " * 22 + "RETAIL INVENTORY EXECUTIVE KPI SUMMARY")
    print("=" * 80)
    print(f" Total SKUs Audited:               {total_skus:>8d}  | Projected Annual Revenue:  ${annual_rev_est:>12,.2f}")
    print(f" Current On-Hand Inventory Units:  {total_stock_units:>8,d}  | Current Inventory Value:   ${total_stock_value:>12,.2f}")
    print(f" Urgent Recommended Reorder Units: {total_reorder_units:>8,d}  | Total Capital Commitment:  ${total_reorder_spend:>12,.2f}")
    print("-" * 80)
    print(f" [!] CRITICAL (Stockout Alert):   {critical_count:>3d} SKUs ({critical_count/total_skus*100:.1f}%)")
    print(f" [*] REORDER NEEDED (Below ROP):   {reorder_count:>3d} SKUs ({reorder_count/total_skus*100:.1f}%)")
    print(f" [V] OPTIMAL (In Safe Range):      {optimal_count:>3d} SKUs ({optimal_count/total_skus*100:.1f}%)")
    print(f" [^] OVERSTOCKED (Excess Capital): {overstock_count:>3d} SKUs ({overstock_count/total_skus*100:.1f}%)")
    print("=" * 80 + "\n")


def print_reorder_recommendation_table(master_df: pd.DataFrame, top_n: int = 15) -> None:
    """Prints a formatted ASCII table of the top urgent reorder recommendations."""
    urgent_df = master_df[master_df["recommended_order_qty"] > 0].copy()
    if urgent_df.empty:
        print("[+] All inventory levels are healthy! No reorders required.")
        return

    top_items = urgent_df.head(top_n)

    print(f"--- Top {len(top_items)} Purchase Order Recommendations (Ranked by Urgency) ---")
    headers = ["SKU", "Product Name", "ABC", "Stock", "ROP", "EOQ", "Runway", "Status", "Order Qty", "Cost ($)"]
    row_format = "{:<9} {:<24} {:<4} {:>5} {:>5} {:>5} {:>7} {:<14} {:>9} {:>10}"

    print("-" * 98)
    print(row_format.format(*headers))
    print("-" * 98)

    for _, row in top_items.iterrows():
        name_trunc = row["product_name"][:23]
        runway_str = f"{row['stockout_runway_days']:.1f}d" if row["stockout_runway_days"] < 999 else ">999d"
        status_tag = f"[{row['inventory_status']}]"
        print(row_format.format(
            row["sku_id"],
            name_trunc,
            row["abc_class"],
            int(row["current_stock"]),
            int(row["reorder_point"]),
            int(row["eoq"]),
            runway_str,
            status_tag,
            int(row["recommended_order_qty"]),
            f"${row['reorder_cost']:,.2f}",
        ))
    print("-" * 98)
    if len(urgent_df) > top_n:
        print(f" ... and {len(urgent_df) - top_n} additional SKUs requiring replenishment.\n")


def print_category_breakdown(master_df: pd.DataFrame) -> None:
    """Prints breakdown of inventory health and reorder spend by category."""
    cat_summary = master_df.groupby("category").agg(
        total_skus=("sku_id", "count"),
        stock_units=("current_stock", "sum"),
        stock_value=("current_stock", lambda s: (s * master_df.loc[s.index, "unit_cost"]).sum()),
        reorder_units=("recommended_order_qty", "sum"),
        reorder_cost=("reorder_cost", "sum"),
        critical_count=("inventory_status", lambda s: (s == "CRITICAL").sum()),
    )

    print("\n--- Category-Level Inventory Breakdown ---")
    headers = ["Category", "SKUs", "Stock Units", "Current Value", "Reorder Units", "Reorder Cost", "Critical"]
    row_format = "{:<24} {:>5} {:>12} {:>14} {:>14} {:>14} {:>9}"
    print("-" * 98)
    print(row_format.format(*headers))
    print("-" * 98)

    for cat, row in cat_summary.iterrows():
        print(row_format.format(
            cat,
            int(row["total_skus"]),
            f"{int(row['stock_units']):,}",
            f"${row['stock_value']:,.2f}",
            f"{int(row['reorder_units']):,}",
            f"${row['reorder_cost']:,.2f}",
            int(row["critical_count"]),
        ))
    print("-" * 98 + "\n")


def export_purchase_orders(
    master_df: pd.DataFrame,
    export_path: str = PO_EXPORT_PATH,
) -> str:
    """
    Exports clean purchase orders for all items requiring reorder to a CSV file.
    """
    orders_df = master_df[master_df["recommended_order_qty"] > 0].copy()
    if orders_df.empty:
        print("[!] No reorder quantities to export.")
        return ""

    orders_df["case_packs_to_order"] = (
        orders_df["recommended_order_qty"] / orders_df["case_pack"]
    ).astype(int)

    export_cols = [
        "sku_id",
        "product_name",
        "category",
        "abc_class",
        "current_stock",
        "incoming_orders",
        "reorder_point",
        "safety_stock",
        "eoq",
        "moq",
        "case_pack",
        "case_packs_to_order",
        "recommended_order_qty",
        "unit_cost",
        "reorder_cost",
        "supplier_lead_time_days",
        "stockout_runway_days",
        "inventory_status",
    ]

    clean_export = orders_df[export_cols].copy()
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    clean_export.to_csv(export_path, index=False)
    print(f"[+] Purchase Order exported successfully: {export_path} ({len(clean_export)} line items)")
    return export_path
