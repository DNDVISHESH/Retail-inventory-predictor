"""Main CLI Entry Point for Retail Inventory & Reorder Predictor."""

import argparse
import sys
import os
import pandas as pd

from src.config import (
    CATALOG_PATH,
    SALES_PATH,
    PO_EXPORT_PATH,
    REPORT_CHART_PATH,
    InventoryConfig,
)
from src.data_generator import generate_retail_dataset
from src.analytics_engine import run_inventory_analysis
from src.simulator import InventorySimulator
from src.visualizer import generate_visual_dashboard
from src.reporter import (
    print_executive_summary,
    print_reorder_recommendation_table,
    print_category_breakdown,
    export_purchase_orders,
)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generates synthetic retail catalog and sales transactions."""
    print("\n========================================================")
    print(" 1. GENERATING SYNTHETIC RETAIL DATASET (NumPy & Pandas)")
    print("========================================================")
    generate_retail_dataset(
        num_skus=args.skus,
        history_days=args.days,
        random_seed=args.seed,
        save_to_disk=True,
    )


def cmd_analyze(args: argparse.Namespace) -> pd.DataFrame:
    """Runs inventory audit, ROP/EOQ optimization, and PO generation."""
    catalog_file = args.catalog or CATALOG_PATH
    sales_file = args.sales or SALES_PATH

    if not os.path.exists(catalog_file) or not os.path.exists(sales_file):
        print(f"[!] Dataset files not found at:\n    Catalog: {catalog_file}\n    Sales: {sales_file}")
        print("[*] Automatically generating realistic retail dataset first...")
        generate_retail_dataset(save_to_disk=True)

    config = InventoryConfig(
        service_level=args.service_level,
        annual_holding_rate=args.holding_rate,
        fixed_order_cost=args.order_cost,
    )

    print(f"\n[*] Loading catalog from: {catalog_file}")
    print(f"[*] Loading sales transactions from: {sales_file}")
    df_catalog = pd.read_csv(catalog_file)
    df_sales = pd.read_csv(sales_file)

    print(f"[*] Running analytical optimization (Service Level: {config.service_level*100:.0f}%, Holding Rate: {config.annual_holding_rate*100:.0f}%, Order Cost: ${config.fixed_order_cost:.2f})...")
    master_df = run_inventory_analysis(df_catalog, df_sales, config)

    # Display reports
    print_executive_summary(master_df)
    print_reorder_recommendation_table(master_df, top_n=args.top_n)
    print_category_breakdown(master_df)

    # Export
    export_path = args.export or PO_EXPORT_PATH
    export_purchase_orders(master_df, export_path)

    return master_df


def cmd_simulate(args: argparse.Namespace, master_df: pd.DataFrame = None) -> dict:
    """Simulates 30-60 day operations comparing Naive vs ROP/EOQ policy."""
    if master_df is None:
        catalog_file = args.catalog or CATALOG_PATH
        sales_file = args.sales or SALES_PATH
        if not os.path.exists(catalog_file) or not os.path.exists(sales_file):
            generate_retail_dataset(save_to_disk=True)
        df_catalog = pd.read_csv(catalog_file)
        df_sales = pd.read_csv(sales_file)
        master_df = run_inventory_analysis(df_catalog, df_sales)

    print("\n========================================================")
    print(" 2. RUNNING REPLENISHMENT POLICY SIMULATION")
    print(f"    Horizon: {args.sim_days} Days | Monte Carlo Seed: {args.seed}")
    print("========================================================")

    simulator = InventorySimulator(master_df, simulation_days=args.sim_days, random_seed=args.seed)
    sim_results = simulator.run_simulation()

    print("\n--- Simulation Results Comparison ---")
    comp_df = sim_results["summary_comparison"]
    col_w = [34, 24, 20, 16]
    row_fmt = "".join(f"{{:<{w}}} " for w in col_w)

    print("-" * 96)
    print(row_fmt.format(*comp_df.columns))
    print("-" * 96)
    for _, row in comp_df.iterrows():
        print(row_fmt.format(str(row["Metric"]), str(row["Predictive (ROP/EOQ)"]), str(row["Naive Periodic"]), str(row["Difference"])))
    print("-" * 96 + "\n")

    return sim_results


def cmd_dashboard(args: argparse.Namespace, master_df: pd.DataFrame = None, sim_results: dict = None) -> str:
    """Generates visual dashboard chart."""
    if master_df is None:
        catalog_file = args.catalog or CATALOG_PATH
        sales_file = args.sales or SALES_PATH
        if not os.path.exists(catalog_file) or not os.path.exists(sales_file):
            generate_retail_dataset(save_to_disk=True)
        df_catalog = pd.read_csv(catalog_file)
        df_sales = pd.read_csv(sales_file)
        master_df = run_inventory_analysis(df_catalog, df_sales)

    if sim_results is None:
        simulator = InventorySimulator(master_df, simulation_days=args.sim_days, random_seed=args.seed)
        sim_results = simulator.run_simulation()

    print("\n========================================================")
    print(" 3. GENERATING VISUAL ANALYTICS DASHBOARD")
    print("========================================================")
    out_path = args.chart_out or REPORT_CHART_PATH
    generate_visual_dashboard(master_df, sim_results, out_path)
    print(f"[+] Visual dashboard saved: {out_path}\n")
    return out_path


def cmd_all(args: argparse.Namespace) -> None:
    """Executes the complete end-to-end pipeline."""
    print("\n=================================================================")
    print("   RETAIL INVENTORY & REORDER PREDICTOR (END-TO-END PIPELINE)   ")
    print("=================================================================")

    # 1. Generate fresh data if requested
    if args.regenerate or not os.path.exists(CATALOG_PATH):
        cmd_generate(args)

    # 2. Analyze & Export POs
    master_df = cmd_analyze(args)

    # 3. Simulate Replenishment Policies
    sim_results = cmd_simulate(args, master_df)

    # 4. Generate Visual Charts
    cmd_dashboard(args, master_df, sim_results)

    print("[SUCCESS] All pipeline stages completed successfully!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retail Inventory & Reorder Predictor using NumPy and Pandas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate synthetic retail dataset")
    p_gen.add_argument("--skus", type=int, default=40, help="Number of SKUs to generate")
    p_gen.add_argument("--days", type=int, default=180, help="Days of sales history")
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed")

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Run inventory health & reorder analysis")
    p_ana.add_argument("--catalog", type=str, default=CATALOG_PATH, help="Catalog CSV path")
    p_ana.add_argument("--sales", type=str, default=SALES_PATH, help="Sales CSV path")
    p_ana.add_argument("--service-level", type=float, default=0.95, help="Target service level (0.85 - 0.99)")
    p_ana.add_argument("--holding-rate", type=float, default=0.20, help="Annual holding cost rate")
    p_ana.add_argument("--order-cost", type=float, default=50.0, help="Fixed cost per purchase order")
    p_ana.add_argument("--top-n", type=int, default=15, help="Top N urgent items to display")
    p_ana.add_argument("--export", type=str, default=PO_EXPORT_PATH, help="Path for purchase orders CSV")

    # simulate
    p_sim = subparsers.add_parser("simulate", help="Run 30-60 day replenishment simulation")
    p_sim.add_argument("--catalog", type=str, default=CATALOG_PATH, help="Catalog CSV path")
    p_sim.add_argument("--sales", type=str, default=SALES_PATH, help="Sales CSV path")
    p_sim.add_argument("--sim-days", type=int, default=45, help="Simulation duration in days")
    p_sim.add_argument("--seed", type=int, default=99, help="Random seed")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Generate Matplotlib visual dashboard PNG")
    p_dash.add_argument("--catalog", type=str, default=CATALOG_PATH, help="Catalog CSV path")
    p_dash.add_argument("--sales", type=str, default=SALES_PATH, help="Sales CSV path")
    p_dash.add_argument("--sim-days", type=int, default=45, help="Simulation duration in days")
    p_dash.add_argument("--seed", type=int, default=99, help="Random seed")
    p_dash.add_argument("--chart-out", type=str, default=REPORT_CHART_PATH, help="Output PNG path")

    # all
    p_all = subparsers.add_parser("all", help="Execute complete pipeline (generate -> analyze -> simulate -> dashboard)")
    p_all.add_argument("--regenerate", action="store_true", help="Force regenerate dataset")
    p_all.add_argument("--skus", type=int, default=40, help="Number of SKUs to generate")
    p_all.add_argument("--days", type=int, default=180, help="Days of sales history")
    p_all.add_argument("--catalog", type=str, default=CATALOG_PATH, help="Catalog CSV path")
    p_all.add_argument("--sales", type=str, default=SALES_PATH, help="Sales CSV path")
    p_all.add_argument("--service-level", type=float, default=0.95, help="Target service level")
    p_all.add_argument("--holding-rate", type=float, default=0.20, help="Annual holding cost rate")
    p_all.add_argument("--order-cost", type=float, default=50.0, help="Fixed cost per purchase order")
    p_all.add_argument("--top-n", type=int, default=15, help="Top N urgent items to display")
    p_all.add_argument("--sim-days", type=int, default=45, help="Simulation duration in days")
    p_all.add_argument("--seed", type=int, default=42, help="Random seed")
    p_all.add_argument("--export", type=str, default=PO_EXPORT_PATH, help="Path for purchase orders CSV")
    p_all.add_argument("--chart-out", type=str, default=REPORT_CHART_PATH, help="Output PNG path")

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        # Default to 'all' if no subcommand provided
        args = parser.parse_args(["all"])
    else:
        args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "all":
        cmd_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
