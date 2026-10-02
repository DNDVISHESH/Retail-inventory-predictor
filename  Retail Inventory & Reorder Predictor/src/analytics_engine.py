"""Core Inventory Analytics & Reorder Prediction Engine using NumPy and Pandas."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from .config import InventoryConfig


def calculate_demand_metrics(
    df_sales: pd.DataFrame,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """
    Calculates rolling demand velocity, volatility (std), and trend using NumPy and Pandas.

    Parameters:
        df_sales: DataFrame containing ['date', 'sku_id', 'units_sold', 'revenue'].
        lookback_days: Window of recent days to evaluate.

    Returns:
        DataFrame indexed by sku_id with columns:
        ['mean_daily_sales', 'std_daily_sales', 'recent_units_sold', 'recent_revenue', 'demand_trend_slope']
    """
    df = df_sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    cutoff_date = max_date - pd.Timedelta(days=lookback_days)
    df_recent = df[df["date"] >= cutoff_date]

    # Group by SKU for basic statistics
    grouped = df_recent.groupby("sku_id")
    agg_stats = grouped.agg(
        mean_daily_sales=("units_sold", "mean"),
        std_daily_sales=("units_sold", "std"),
        recent_units_sold=("units_sold", "sum"),
        recent_revenue=("revenue", "sum"),
    )

    # Standard deviation fallback to 0 if NaN (single observation or all zero)
    agg_stats["std_daily_sales"] = agg_stats["std_daily_sales"].fillna(0.0)

    # Compute vectorized demand trend slope (units per day) using NumPy least squares
    trends = {}
    for sku_id, group in df_recent.groupby("sku_id"):
        # Sort by date
        sorted_g = group.sort_values("date")
        y = sorted_g["units_sold"].to_numpy()
        n = len(y)
        if n >= 2:
            x = np.arange(n)
            # Slope m = Cov(x, y) / Var(x)
            x_bar = np.mean(x)
            y_bar = np.mean(y)
            var_x = np.sum((x - x_bar) ** 2)
            if var_x > 0:
                slope = float(np.sum((x - x_bar) * (y - y_bar)) / var_x)
            else:
                slope = 0.0
        else:
            slope = 0.0
        trends[sku_id] = slope

    agg_stats["demand_trend_slope"] = pd.Series(trends)
    return agg_stats


def calculate_seasonality_indices(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates day-of-week seasonality index for each SKU.
    Index > 1.0 indicates higher than average demand for that weekday.
    """
    df = df_sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Pivot mean units sold by SKU and DOW
    pivot = df.pivot_table(index="sku_id", columns="day_of_week", values="units_sold", aggfunc="mean").fillna(0.0)
    # Ensure all days present
    for d in day_order:
        if d not in pivot.columns:
            pivot[d] = 0.0
    pivot = pivot[day_order]

    # Normalize by each SKU's overall average
    overall_mean = pivot.mean(axis=1).replace(0, np.nan)
    seasonality_df = pivot.div(overall_mean, axis=0).fillna(1.0)
    seasonality_df.columns = [f"dow_{d[:3].lower()}_idx" for d in seasonality_df.columns]
    return seasonality_df


def calculate_safety_stock(
    lead_time_days: np.ndarray,
    lead_time_std_days: np.ndarray,
    mean_daily_sales: np.ndarray,
    std_daily_sales: np.ndarray,
    z_score: float,
) -> np.ndarray:
    """
    Calculates dynamic safety stock factoring both demand volatility and supplier lead-time variance:
    SS = Z * sqrt( L * sigma_d^2 + mu_d^2 * sigma_L^2 )
    """
    term_demand_var = lead_time_days * (std_daily_sales ** 2)
    term_lead_var = (mean_daily_sales ** 2) * (lead_time_std_days ** 2)
    total_var = np.maximum(0.0, term_demand_var + term_lead_var)
    safety_stock = np.ceil(z_score * np.sqrt(total_var)).astype(int)
    return np.maximum(0, safety_stock)


def calculate_reorder_point(
    mean_daily_sales: np.ndarray,
    lead_time_days: np.ndarray,
    safety_stock: np.ndarray,
) -> np.ndarray:
    """
    Calculates Reorder Point (ROP):
    ROP = (mu_d * L) + SS
    """
    lead_time_demand = mean_daily_sales * lead_time_days
    rop = np.ceil(lead_time_demand + safety_stock).astype(int)
    return np.maximum(0, rop)


def calculate_eoq(
    annual_demand: np.ndarray,
    fixed_order_cost: float,
    holding_cost_per_unit_year: np.ndarray,
    moq: np.ndarray,
    case_pack: np.ndarray,
) -> np.ndarray:
    """
    Calculates Economic Order Quantity (EOQ) using Wilson's formula, bounded by MOQ and case pack:
    EOQ = sqrt( (2 * D * S) / H )
    """
    safe_h = np.maximum(0.001, holding_cost_per_unit_year)
    raw_eoq = np.sqrt((2.0 * np.maximum(0.0, annual_demand) * fixed_order_cost) / safe_h)
    
    # Batch up to case pack multiples
    safe_pack = np.maximum(1, case_pack)
    batched_eoq = np.ceil(raw_eoq / safe_pack) * safe_pack
    # Apply MOQ constraint
    final_eoq = np.maximum(moq, batched_eoq).astype(int)
    return final_eoq


def classify_abc(revenue_series: pd.Series) -> pd.Series:
    """
    Classifies SKUs into ABC categories using Pareto 80/15/5 rule based on cumulative revenue.
    Class A: Top ~80% of revenue
    Class B: Next ~15% of revenue (80% - 95%)
    Class C: Bottom ~5% of revenue (95% - 100%)
    """
    sorted_rev = revenue_series.sort_values(ascending=False)
    total_rev = sorted_rev.sum()
    if total_rev <= 0:
        return pd.Series("C", index=revenue_series.index)

    cum_share = sorted_rev.cumsum() / total_rev
    classes = pd.Series(index=sorted_rev.index, dtype=str)

    classes[cum_share <= 0.80] = "A"
    classes[(cum_share > 0.80) & (cum_share <= 0.95)] = "B"
    classes[cum_share > 0.95] = "C"

    # Ensure at least the single highest item gets 'A' if 80% threshold had no match
    if (classes == "A").sum() == 0 and len(classes) > 0:
        classes.iloc[0] = "A"

    return classes.reindex(revenue_series.index)


def evaluate_inventory_health(
    df: pd.DataFrame,
    critical_runway_days: float = 7.0,
    warning_runway_days: float = 14.0,
) -> pd.DataFrame:
    """
    Vectorized evaluation of inventory runway, health status, and reorder urgency.
    """
    # Days of runway = current_stock / mean_daily_sales
    stock = df["current_stock"].to_numpy().astype(float)
    mu_d = df["mean_daily_sales"].to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        runway = np.where(mu_d > 0.001, stock / mu_d, 999.0)
    df["stockout_runway_days"] = np.round(runway, 1)

    # Effective stock = current_stock + incoming_orders
    effective_stock = df["current_stock"] + df["incoming_orders"]
    df["effective_stock"] = effective_stock

    # Categorize status:
    # 1. CRITICAL: current_stock <= safety_stock
    # 2. REORDER NEEDED: effective_stock <= rop (and not critical)
    # 3. OPTIMAL: rop < effective_stock <= rop + eoq
    # 4. OVERSTOCKED: effective_stock > rop + eoq
    conditions = [
        df["current_stock"] <= df["safety_stock"],
        effective_stock <= df["reorder_point"],
        effective_stock <= (df["reorder_point"] + df["eoq"]),
    ]
    choices = ["CRITICAL", "REORDER NEEDED", "OPTIMAL"]
    df["inventory_status"] = np.select(conditions, choices, default="OVERSTOCKED")

    # Recommended Order Quantity
    # Deficit = max(0, ROP + EOQ - effective_stock) when effective_stock <= ROP
    needs_order = effective_stock <= df["reorder_point"]
    deficit = np.maximum(0, (df["reorder_point"] + df["eoq"]) - effective_stock)
    
    # Batch to case pack
    case_pack = np.maximum(1, df["case_pack"].to_numpy())
    batched_deficit = np.ceil(deficit / case_pack) * case_pack
    recommended_qty = np.where(needs_order, np.maximum(df["moq"], batched_deficit), 0).astype(int)
    df["recommended_order_qty"] = recommended_qty

    # Order Cost Commitment
    df["reorder_cost"] = np.round(df["recommended_order_qty"] * df["unit_cost"], 2)

    # Reorder Priority Score: (ROP - Effective Stock) / ROP adjusted by ABC priority weight
    abc_weights = {"A": 3.0, "B": 2.0, "C": 1.0}
    weight_col = df["abc_class"].map(abc_weights).fillna(1.0).to_numpy()
    rop_arr = np.maximum(1, df["reorder_point"].to_numpy().astype(float))
    urgency_ratio = np.maximum(0.0, (rop_arr - effective_stock.to_numpy().astype(float)) / rop_arr)
    df["priority_score"] = np.round(urgency_ratio * weight_col * 100.0, 1)

    return df


def run_inventory_analysis(
    df_catalog: pd.DataFrame,
    df_sales: pd.DataFrame,
    config: Optional[InventoryConfig] = None,
) -> pd.DataFrame:
    """
    Orchestrates end-to-end inventory analysis and reorder predictions.

    Returns:
        Master analysis DataFrame sorted by reorder urgency (priority_score descending).
    """
    if config is None:
        config = InventoryConfig()

    # 1. Compute Demand Velocity & Statistics
    demand_df = calculate_demand_metrics(df_sales, lookback_days=config.demand_lookback_days)
    
    # Merge catalog with demand metrics
    merged = df_catalog.merge(demand_df, on="sku_id", how="left")
    merged["mean_daily_sales"] = merged["mean_daily_sales"].fillna(0.0)
    merged["std_daily_sales"] = merged["std_daily_sales"].fillna(0.0)
    merged["recent_units_sold"] = merged["recent_units_sold"].fillna(0)
    merged["recent_revenue"] = merged["recent_revenue"].fillna(0.0)
    merged["demand_trend_slope"] = merged["demand_trend_slope"].fillna(0.0)

    # 2. Annualized Demand & Holding Cost
    annual_demand = merged["mean_daily_sales"].to_numpy() * config.annual_operating_days
    merged["annual_demand"] = np.round(annual_demand, 0).astype(int)
    holding_cost_unit = merged["unit_cost"].to_numpy() * config.annual_holding_rate
    merged["annual_holding_cost_per_unit"] = np.round(holding_cost_unit, 2)

    # 3. Dynamic Safety Stock (NumPy vectorized)
    ss = calculate_safety_stock(
        lead_time_days=merged["supplier_lead_time_days"].to_numpy(),
        lead_time_std_days=merged["lead_time_std_days"].to_numpy(),
        mean_daily_sales=merged["mean_daily_sales"].to_numpy(),
        std_daily_sales=merged["std_daily_sales"].to_numpy(),
        z_score=config.z_score,
    )
    merged["safety_stock"] = ss

    # 4. Reorder Point (ROP)
    rop = calculate_reorder_point(
        mean_daily_sales=merged["mean_daily_sales"].to_numpy(),
        lead_time_days=merged["supplier_lead_time_days"].to_numpy(),
        safety_stock=ss,
    )
    merged["reorder_point"] = rop

    # 5. Economic Order Quantity (EOQ)
    eoq = calculate_eoq(
        annual_demand=annual_demand,
        fixed_order_cost=config.fixed_order_cost,
        holding_cost_per_unit_year=holding_cost_unit,
        moq=merged["moq"].to_numpy(),
        case_pack=merged["case_pack"].to_numpy(),
    )
    merged["eoq"] = eoq

    # 6. ABC Pareto Analysis (based on annual projected revenue)
    projected_annual_rev = merged["annual_demand"] * merged["selling_price"]
    merged["annual_projected_revenue"] = np.round(projected_annual_rev, 2)
    merged["abc_class"] = classify_abc(merged["annual_projected_revenue"])

    # 7. Inventory Health, Runway & Purchase Order Recommendations
    master_df = evaluate_inventory_health(
        merged,
        critical_runway_days=config.critical_runway_days,
        warning_runway_days=config.warning_runway_days,
    )

    # Sort by priority score descending
    master_df = master_df.sort_values(by=["priority_score", "stockout_runway_days"], ascending=[False, True])
    return master_df
