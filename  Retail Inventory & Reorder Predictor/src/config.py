"""Configuration parameters for Retail Inventory & Reorder Predictor."""

import os
from dataclasses import dataclass
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

CATALOG_PATH = os.path.join(DATA_DIR, "inventory_catalog.csv")
SALES_PATH = os.path.join(DATA_DIR, "sales_history.csv")
PO_EXPORT_PATH = os.path.join(DATA_DIR, "purchase_orders.csv")
REPORT_CHART_PATH = os.path.join(REPORTS_DIR, "inventory_analytics.png")

# Common Service Level Z-Scores (Standard Normal Inverse CDF)
Z_SCORE_MAP: Dict[float, float] = {
    0.85: 1.036,
    0.90: 1.282,
    0.95: 1.645,
    0.98: 2.054,
    0.99: 2.326,
    0.999: 3.090,
}


@dataclass
class InventoryConfig:
    """Configurable economic and operational parameters."""

    service_level: float = 0.95
    annual_holding_rate: float = 0.20  # 20% annual inventory holding cost
    fixed_order_cost: float = 50.0     # $50 administrative/shipping cost per PO
    annual_operating_days: int = 365
    demand_lookback_days: int = 30     # Window for rolling average demand
    trend_lookback_days: int = 60      # Window for linear trend calculation
    critical_runway_days: float = 7.0  # Under 7 days runway is marked critical
    warning_runway_days: float = 14.0  # Under 14 days runway is marked warning

    @property
    def z_score(self) -> float:
        """Returns the Z-score for the configured service level."""
        if self.service_level in Z_SCORE_MAP:
            return Z_SCORE_MAP[self.service_level]
        # Approximate if custom value provided:
        # Abramowitz and Stegun / Hastings rational approximation for normal quantile
        p = self.service_level
        t = (-2.0 * 0.5 * (1.0 - p)) ** 0.5 if p < 0.5 else (-2.0 * (1.0 - p)) ** 0.5
        # Fallback to closest known key
        closest_key = min(Z_SCORE_MAP.keys(), key=lambda k: abs(k - self.service_level))
        return Z_SCORE_MAP[closest_key]
