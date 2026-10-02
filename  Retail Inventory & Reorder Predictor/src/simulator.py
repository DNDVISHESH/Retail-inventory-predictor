"""Inventory Replenishment Policy Simulator comparing Naive vs ROP/EOQ predictive replenishment."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from .config import InventoryConfig


class InventorySimulator:
    """
    Simulates daily retail inventory operations over a future horizon.
    Compares:
    1. Naive Periodic Replenishment (orders fixed quantity every 14 days)
    2. Predictive ROP/EOQ Replenishment (dynamically triggers EOQ when inventory drops <= ROP)
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        simulation_days: int = 45,
        config: InventoryConfig = None,
        random_seed: int = 99,
    ):
        self.master_df = master_df.copy()
        self.days = simulation_days
        self.config = config or InventoryConfig()
        self.seed = random_seed

    def run_simulation(self) -> Dict[str, Any]:
        np.random.seed(self.seed)
        num_skus = len(self.master_df)

        sku_ids = self.master_df["sku_id"].to_numpy()
        unit_costs = self.master_df["unit_cost"].to_numpy()
        selling_prices = self.master_df["selling_price"].to_numpy()
        mean_demands = self.master_df["mean_daily_sales"].to_numpy()
        std_demands = self.master_df["std_daily_sales"].to_numpy()
        lead_means = self.master_df["supplier_lead_time_days"].to_numpy()
        lead_stds = self.master_df["lead_time_std_days"].to_numpy()
        initial_stocks = self.master_df["current_stock"].to_numpy()
        rops = self.master_df["reorder_point"].to_numpy()
        eoqs = self.master_df["eoq"].to_numpy()
        moqs = self.master_df["moq"].to_numpy()
        case_packs = self.master_df["case_pack"].to_numpy()

        # Generate daily future customer demand for all SKUs across all days
        # Shape: (simulation_days, num_skus)
        # Using Poisson/normal distribution based on historical mean and std
        future_demand = np.zeros((self.days, num_skus), dtype=int)
        for i in range(num_skus):
            mu = mean_demands[i]
            sigma = max(0.5, std_demands[i])
            daily = np.random.normal(loc=mu, scale=sigma, size=self.days)
            future_demand[:, i] = np.maximum(0, np.round(daily)).astype(int)

        # -------------------------------------------------------------
        # 1. Run Predictive ROP/EOQ Policy
        # -------------------------------------------------------------
        res_predictive = self._simulate_policy(
            policy_name="Predictive ROP/EOQ",
            num_skus=num_skus,
            future_demand=future_demand,
            initial_stocks=initial_stocks.copy(),
            rops=rops,
            eoqs=eoqs,
            moqs=moqs,
            case_packs=case_packs,
            lead_means=lead_means,
            lead_stds=lead_stds,
            unit_costs=unit_costs,
            selling_prices=selling_prices,
            is_predictive=True,
        )

        # -------------------------------------------------------------
        # 2. Run Naive Periodic Policy (every 14 days, order fixed 2-week supply)
        # -------------------------------------------------------------
        res_naive = self._simulate_policy(
            policy_name="Naive Periodic (14-Day)",
            num_skus=num_skus,
            future_demand=future_demand,
            initial_stocks=initial_stocks.copy(),
            rops=rops,
            eoqs=eoqs,
            moqs=moqs,
            case_packs=case_packs,
            lead_means=lead_means,
            lead_stds=lead_stds,
            unit_costs=unit_costs,
            selling_prices=selling_prices,
            is_predictive=False,
            naive_interval=14,
        )

        return {
            "simulation_days": self.days,
            "num_skus": num_skus,
            "predictive": res_predictive,
            "naive": res_naive,
            "summary_comparison": self._create_summary_comparison(res_predictive, res_naive),
        }

    def _simulate_policy(
        self,
        policy_name: str,
        num_skus: int,
        future_demand: np.ndarray,
        initial_stocks: np.ndarray,
        rops: np.ndarray,
        eoqs: np.ndarray,
        moqs: np.ndarray,
        case_packs: np.ndarray,
        lead_means: np.ndarray,
        lead_stds: np.ndarray,
        unit_costs: np.ndarray,
        selling_prices: np.ndarray,
        is_predictive: bool,
        naive_interval: int = 14,
    ) -> Dict[str, Any]:
        # Track pipeline orders: list of (arrival_day, sku_idx, quantity)
        pipeline_orders: List[Dict[str, Any]] = []

        on_hand_history = np.zeros((self.days, num_skus))
        lost_sales_history = np.zeros((self.days, num_skus))
        units_sold_history = np.zeros((self.days, num_skus))
        orders_placed_count = 0
        total_order_spend = 0.0

        current_stock = initial_stocks.copy()

        for day in range(self.days):
            # 1. Receive incoming shipments scheduled to arrive on or before today
            remaining_orders = []
            for order in pipeline_orders:
                if order["arrival_day"] <= day:
                    current_stock[order["sku_idx"]] += order["qty"]
                else:
                    remaining_orders.append(order)
            pipeline_orders = remaining_orders

            # 2. Calculate on-order quantity per SKU
            on_order = np.zeros(num_skus, dtype=int)
            for order in pipeline_orders:
                on_order[order["sku_idx"]] += order["qty"]

            # 3. Check replenishment triggers and place orders
            if is_predictive:
                # Predictive: Trigger order when (on_hand + on_order) <= ROP
                effective_stock = current_stock + on_order
                needs_order = effective_stock <= rops
                for i in np.where(needs_order)[0]:
                    deficit = (rops[i] + eoqs[i]) - effective_stock[i]
                    batched = int(np.ceil(deficit / case_packs[i]) * case_packs[i])
                    order_qty = max(int(moqs[i]), batched)
                    
                    # Lead time with variance
                    sampled_lead = max(1, int(np.round(np.random.normal(lead_means[i], lead_stds[i]))))
                    pipeline_orders.append({
                        "arrival_day": day + sampled_lead,
                        "sku_idx": i,
                        "qty": order_qty,
                    })
                    orders_placed_count += 1
                    total_order_spend += order_qty * unit_costs[i]
            else:
                # Naive: Place order periodically every `naive_interval` days
                if day % naive_interval == 0:
                    for i in range(num_skus):
                        # Naive order: 14 days of average demand
                        target_qty = int(np.ceil(lead_means[i] * 2 * (eoqs[i] / 20.0)))
                        order_qty = max(int(moqs[i]), target_qty)
                        sampled_lead = max(1, int(np.round(np.random.normal(lead_means[i], lead_stds[i]))))
                        pipeline_orders.append({
                            "arrival_day": day + sampled_lead,
                            "sku_idx": i,
                            "qty": order_qty,
                        })
                        orders_placed_count += 1
                        total_order_spend += order_qty * unit_costs[i]

            # 4. Fulfill customer demand
            demand_today = future_demand[day, :]
            fulfilled = np.minimum(current_stock, demand_today)
            lost = demand_today - fulfilled

            current_stock -= fulfilled

            on_hand_history[day, :] = current_stock
            units_sold_history[day, :] = fulfilled
            lost_sales_history[day, :] = lost

        # Compute aggregate metrics
        total_demand = int(future_demand.sum())
        total_sold = int(units_sold_history.sum())
        total_lost_units = int(lost_sales_history.sum())
        fill_rate = (total_sold / max(1, total_demand)) * 100.0

        # Stockout days: SKU-days where stock reached 0
        stockout_sku_days = int((on_hand_history == 0).sum())
        total_sku_days = self.days * num_skus
        stockout_pct = (stockout_sku_days / total_sku_days) * 100.0

        # Lost revenue
        lost_revenue = float((lost_sales_history * selling_prices).sum())
        actual_revenue = float((units_sold_history * selling_prices).sum())

        # Average inventory value & holding cost
        avg_on_hand_units = float(on_hand_history.mean())
        daily_inv_value = on_hand_history * unit_costs
        avg_inv_value = float(daily_inv_value.sum(axis=1).mean())
        holding_cost = avg_inv_value * self.config.annual_holding_rate * (self.days / self.config.annual_operating_days)
        ordering_cost = orders_placed_count * self.config.fixed_order_cost
        total_operational_cost = holding_cost + ordering_cost + (lost_revenue * 0.25)  # penalty for lost goodwill

        return {
            "policy_name": policy_name,
            "total_demand": total_demand,
            "total_units_sold": total_sold,
            "total_lost_units": total_lost_units,
            "fill_rate_pct": round(fill_rate, 2),
            "stockout_rate_pct": round(stockout_pct, 2),
            "lost_revenue": round(lost_revenue, 2),
            "actual_revenue": round(actual_revenue, 2),
            "orders_placed": orders_placed_count,
            "average_inventory_value": round(avg_inv_value, 2),
            "holding_cost": round(holding_cost, 2),
            "ordering_cost": round(ordering_cost, 2),
            "total_cost": round(total_operational_cost, 2),
            "on_hand_timeline": on_hand_history.sum(axis=1).tolist(),
        }

    def _create_summary_comparison(self, pred: Dict[str, Any], naive: Dict[str, Any]) -> pd.DataFrame:
        metrics = [
            ("Service Level / Fill Rate (%)", f"{pred['fill_rate_pct']}%", f"{naive['fill_rate_pct']}%", f"{pred['fill_rate_pct'] - naive['fill_rate_pct']:+.1f}%"),
            ("Stockout SKU-Days (%)", f"{pred['stockout_rate_pct']}%", f"{naive['stockout_rate_pct']}%", f"{pred['stockout_rate_pct'] - naive['stockout_rate_pct']:+.1f}%"),
            ("Lost Revenue ($)", f"${pred['lost_revenue']:,.2f}", f"${naive['lost_revenue']:,.2f}", f"${pred['lost_revenue'] - naive['lost_revenue']:,.2f}"),
            ("Average Inventory Value ($)", f"${pred['average_inventory_value']:,.2f}", f"${naive['average_inventory_value']:,.2f}", f"${pred['average_inventory_value'] - naive['average_inventory_value']:,.2f}"),
            ("Holding Cost ($)", f"${pred['holding_cost']:,.2f}", f"${naive['holding_cost']:,.2f}", f"${pred['holding_cost'] - naive['holding_cost']:,.2f}"),
            ("Purchase Orders Placed", f"{pred['orders_placed']}", f"{naive['orders_placed']}", f"{pred['orders_placed'] - naive['orders_placed']:+d}"),
            ("Total Operational Cost ($)", f"${pred['total_cost']:,.2f}", f"${naive['total_cost']:,.2f}", f"${pred['total_cost'] - naive['total_cost']:,.2f}"),
        ]
        return pd.DataFrame(metrics, columns=["Metric", "Predictive (ROP/EOQ)", "Naive Periodic", "Difference"])
