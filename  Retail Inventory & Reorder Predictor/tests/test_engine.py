"""Unit tests for the Inventory Analytics Engine formulas and logic."""

import unittest
import numpy as np
import pandas as pd
from src.config import InventoryConfig
from src.analytics_engine import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    classify_abc,
    evaluate_inventory_health,
)


class TestInventoryEngine(unittest.TestCase):
    """Test suite verifying mathematical correctness of inventory calculations."""

    def test_safety_stock_formula(self):
        # L = 9, sigma_d = 4, mu_d = 10, sigma_L = 1, Z = 1.645
        # Total Var = 9*(16) + 100*(1) = 144 + 100 = 244
        # sqrt(244) ≈ 15.6205
        # SS = ceil(1.645 * 15.6205) = ceil(25.695) = 26
        L = np.array([9.0])
        sigma_L = np.array([1.0])
        mu_d = np.array([10.0])
        sigma_d = np.array([4.0])
        z = 1.645

        ss = calculate_safety_stock(L, sigma_L, mu_d, sigma_d, z)
        self.assertEqual(ss[0], 26)

    def test_reorder_point(self):
        mu_d = np.array([15.0])
        L = np.array([4.0])
        ss = np.array([20])

        # ROP = ceil(15 * 4 + 20) = 80
        rop = calculate_reorder_point(mu_d, L, ss)
        self.assertEqual(rop[0], 80)

    def test_eoq_wilson_formula(self):
        # D = 1000, S = 50, H = 2.50
        # raw_eoq = sqrt( (2 * 1000 * 50) / 2.50 ) = sqrt(40000) = 200
        # MOQ = 50, case_pack = 10
        annual_demand = np.array([1000.0])
        fixed_s = 50.0
        h = np.array([2.50])
        moq = np.array([50])
        case_pack = np.array([10])

        eoq = calculate_eoq(annual_demand, fixed_s, h, moq, case_pack)
        self.assertEqual(eoq[0], 200)

    def test_eoq_moq_bound(self):
        # If calculated EOQ is 200, but supplier MOQ is 300, EOQ must elevate to 300
        annual_demand = np.array([1000.0])
        fixed_s = 50.0
        h = np.array([2.50])
        moq = np.array([300])
        case_pack = np.array([10])

        eoq = calculate_eoq(annual_demand, fixed_s, h, moq, case_pack)
        self.assertEqual(eoq[0], 300)

    def test_eoq_case_pack_rounding(self):
        # raw_eoq = 205, case pack = 24 -> ceil(205/24)*24 = 9*24 = 216
        annual_demand = np.array([1050.0])
        fixed_s = 50.0
        h = np.array([2.50])
        moq = np.array([10])
        case_pack = np.array([24])

        eoq = calculate_eoq(annual_demand, fixed_s, h, moq, case_pack)
        self.assertEqual(eoq[0] % 24, 0)
        self.assertGreaterEqual(eoq[0], 205)

    def test_abc_pareto_classification(self):
        rev = pd.Series([80000.0, 15000.0, 5000.0], index=["SKU-1", "SKU-2", "SKU-3"])
        classes = classify_abc(rev)
        self.assertEqual(classes["SKU-1"], "A")
        self.assertEqual(classes["SKU-2"], "B")
        self.assertEqual(classes["SKU-3"], "C")

    def test_inventory_health_statuses(self):
        sample = pd.DataFrame({
            "sku_id": ["SKU-1", "SKU-2", "SKU-3", "SKU-4"],
            "product_name": ["P1", "P2", "P3", "P4"],
            "current_stock": [10, 45, 90, 300],
            "incoming_orders": [0, 0, 0, 0],
            "safety_stock": [20, 20, 20, 20],
            "reorder_point": [50, 50, 50, 50],
            "eoq": [100, 100, 100, 100],
            "moq": [10, 10, 10, 10],
            "case_pack": [5, 5, 5, 5],
            "unit_cost": [10.0, 10.0, 10.0, 10.0],
            "mean_daily_sales": [5.0, 5.0, 5.0, 5.0],
            "abc_class": ["A", "B", "C", "C"],
        })

        evaluated = evaluate_inventory_health(sample)
        # SKU-1: stock 10 <= SS 20 -> CRITICAL
        self.assertEqual(evaluated.loc[0, "inventory_status"], "CRITICAL")
        self.assertGreater(evaluated.loc[0, "recommended_order_qty"], 0)

        # SKU-2: SS 20 < stock 45 <= ROP 50 -> REORDER NEEDED
        self.assertEqual(evaluated.loc[1, "inventory_status"], "REORDER NEEDED")
        self.assertGreater(evaluated.loc[1, "recommended_order_qty"], 0)

        # SKU-3: ROP 50 < stock 90 <= ROP+EOQ (150) -> OPTIMAL
        self.assertEqual(evaluated.loc[2, "inventory_status"], "OPTIMAL")
        self.assertEqual(evaluated.loc[2, "recommended_order_qty"], 0)

        # SKU-4: stock 300 > ROP+EOQ (150) -> OVERSTOCKED
        self.assertEqual(evaluated.loc[3, "inventory_status"], "OVERSTOCKED")
        self.assertEqual(evaluated.loc[3, "recommended_order_qty"], 0)

    def test_zero_demand_handling(self):
        sample = pd.DataFrame({
            "sku_id": ["SKU-ZERO"],
            "product_name": ["Dormant Product"],
            "current_stock": [50],
            "incoming_orders": [0],
            "safety_stock": [0],
            "reorder_point": [0],
            "eoq": [10],
            "moq": [10],
            "case_pack": [1],
            "unit_cost": [5.0],
            "mean_daily_sales": [0.0],
            "abc_class": ["C"],
        })
        evaluated = evaluate_inventory_health(sample)
        self.assertEqual(evaluated.loc[0, "stockout_runway_days"], 999.0)
        self.assertEqual(evaluated.loc[0, "inventory_status"], "OVERSTOCKED")


if __name__ == "__main__":
    unittest.main()
