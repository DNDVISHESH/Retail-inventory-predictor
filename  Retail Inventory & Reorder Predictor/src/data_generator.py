"""Realistic Synthetic Retail Dataset Generator using NumPy and Pandas."""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from .config import CATALOG_PATH, SALES_PATH


def generate_retail_dataset(
    num_skus: int = 40,
    history_days: int = 180,
    end_date_str: str = "2026-09-18",
    random_seed: int = 42,
    save_to_disk: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a realistic multi-category retail product catalog and daily sales history.

    Parameters:
        num_skus: Number of distinct product SKUs to generate.
        history_days: Number of historical days of sales data to simulate.
        end_date_str: End date for the sales history.
        random_seed: Seed for reproducible NumPy random generation.
        save_to_disk: If True, saves files to data/ directory.

    Returns:
        Tuple of (df_catalog, df_sales).
    """
    np.random.seed(random_seed)

    # Categories and sample product name templates
    catalog_templates = [
        # Electronics (High value, longer lead time)
        ("Electronics", "Wireless Bluetooth Earbuds", 32.0, 79.99, 14, 2.5, 20, 10),
        ("Electronics", "USB-C Fast Charging Cable 2m", 3.5, 14.99, 7, 1.2, 50, 25),
        ("Electronics", "Ergonomic Mechanical Keyboard", 45.0, 119.99, 18, 3.0, 15, 5),
        ("Electronics", "Smart Fitness Tracker Band", 28.0, 69.99, 12, 2.0, 25, 10),
        ("Electronics", "Noise-Cancelling Over-Ear Headphones", 75.0, 189.99, 21, 3.5, 10, 5),
        ("Electronics", "4K Ultra HD Streaming Stick", 19.0, 49.99, 10, 1.8, 30, 10),
        ("Electronics", "Portable Power Bank 20000mAh", 16.0, 39.99, 14, 2.2, 40, 20),
        ("Electronics", "Wireless Qi Desktop Charger", 11.0, 27.99, 9, 1.5, 30, 15),

        # Groceries (Fast moving, low margin, short lead time)
        ("Groceries", "Organic Cold Pressed Olive Oil 500ml", 5.2, 11.49, 4, 0.8, 48, 12),
        ("Groceries", "Artisan Roasted Coffee Beans 1kg", 8.5, 18.99, 5, 1.0, 36, 12),
        ("Groceries", "Whole Grain Rolled Oats 1kg", 1.4, 3.99, 3, 0.5, 60, 20),
        ("Groceries", "Pure Almond Milk Unsweetened 1L", 1.2, 3.49, 4, 0.6, 72, 24),
        ("Groceries", "Premium Dark Chocolate Bar 85%", 1.6, 4.29, 5, 0.7, 50, 25),
        ("Groceries", "Natural Greek Yogurt 500g", 1.8, 4.49, 3, 0.4, 48, 12),
        ("Groceries", "Grade-A Wildflower Honey 400g", 4.0, 9.99, 6, 1.0, 36, 12),
        ("Groceries", "Organic Green Tea Bags 50ct", 2.2, 6.49, 5, 0.8, 40, 20),

        # Apparel (Seasonal, mid margin, moderate lead time)
        ("Apparel", "Cotton Classic Crewneck T-Shirt", 4.8, 19.99, 12, 2.0, 50, 25),
        ("Apparel", "Slim-Fit Stretch Denim Jeans", 16.5, 54.99, 16, 2.8, 30, 10),
        ("Apparel", "Breathable Athletic Running Socks 3pk", 2.5, 12.99, 10, 1.5, 60, 20),
        ("Apparel", "Waterproof Lightweight Windbreaker", 22.0, 69.99, 18, 3.0, 20, 10),
        ("Apparel", "Wool Blend Winter Knit Beanie", 4.0, 16.99, 14, 2.0, 40, 20),
        ("Apparel", "Casual Canvas Everyday Sneakers", 18.0, 59.99, 20, 3.5, 24, 12),
        ("Apparel", "Thermal Base Layer Leggings", 7.5, 24.99, 15, 2.2, 30, 15),
        ("Apparel", "Polarized UV400 Sunglasses", 6.0, 29.99, 11, 1.8, 30, 10),

        # Home & Kitchen (Moderate moving, steady demand)
        ("Home & Kitchen", "Stainless Steel Non-Stick Skillet 28cm", 14.0, 39.99, 12, 2.0, 20, 10),
        ("Home & Kitchen", "Double-Walled Insulated Tumbler 750ml", 5.5, 19.99, 8, 1.2, 40, 20),
        ("Home & Kitchen", "Microfiber Cleaning Towels 12pk", 3.2, 12.99, 6, 1.0, 60, 20),
        ("Home & Kitchen", "Bamboo Cutting Board 3-Piece Set", 8.0, 26.99, 14, 2.5, 24, 12),
        ("Home & Kitchen", "Scented Soy Wax Candle Jar", 3.8, 14.99, 9, 1.4, 36, 12),
        ("Home & Kitchen", "Silicone Cooking Utensil 8-Piece Set", 6.5, 22.99, 11, 1.8, 25, 5),
        ("Home & Kitchen", "Glass Airtight Meal Prep Containers 4pk", 9.0, 29.99, 13, 2.0, 20, 10),
        ("Home & Kitchen", "Memory Foam Ergonomic Seat Cushion", 11.5, 34.99, 15, 2.5, 20, 10),

        # Health & Personal Care (Fast/steady moving, high repeat purchase)
        ("Health & Personal Care", "Daily Multivitamin Gummies 90ct", 4.5, 15.99, 7, 1.2, 48, 24),
        ("Health & Personal Care", "Moisturizing Ceramide Facial Cleanser", 5.0, 16.99, 8, 1.5, 40, 20),
        ("Health & Personal Care", "Mineral Sunscreen SPF 50 Broad Spectrum", 4.2, 14.49, 7, 1.0, 50, 25),
        ("Health & Personal Care", "Natural Tea Tree Oil Shampoo 400ml", 3.5, 11.99, 6, 1.0, 48, 12),
        ("Health & Personal Care", "Sonic Electric Toothbrush Replacement Heads", 4.0, 18.99, 10, 1.8, 40, 20),
        ("Health & Personal Care", "Hand Sanitizer Gel 500ml Dispenser", 1.8, 5.99, 4, 0.5, 60, 30),
        ("Health & Personal Care", "Lavender Aromatherapy Essential Oil 30ml", 3.0, 12.49, 8, 1.2, 40, 20),
        ("Health & Personal Care", "Deep Tissue Foam Muscle Roller", 7.0, 24.99, 12, 2.0, 20, 10),
    ]

    selected_templates = catalog_templates[:num_skus]
    skus = []
    base_demands = []

    # Assign base demand distributions across items (some power-law fast movers, some long-tail)
    for i, item in enumerate(selected_templates):
        sku_code = f"SKU-{1001 + i}"
        cat, name, cost, price, lead_time, lead_std, moq, pack = item

        # Category multiplier for base daily demand
        if cat == "Groceries":
            base_d = np.random.uniform(15.0, 55.0)
        elif cat == "Health & Personal Care":
            base_d = np.random.uniform(8.0, 30.0)
        elif cat == "Home & Kitchen":
            base_d = np.random.uniform(4.0, 18.0)
        elif cat == "Apparel":
            base_d = np.random.uniform(3.0, 15.0)
        else:  # Electronics
            base_d = np.random.uniform(2.0, 12.0)

        base_demands.append(base_d)

        # Simulate current stock condition:
        # Intentionally inject diverse stock health (some critically low, some reorder-needed, some healthy, some overstocked)
        stock_scenario = np.random.choice(["critical", "reorder", "healthy", "overstocked"], p=[0.20, 0.30, 0.35, 0.15])
        expected_rop = base_d * lead_time + 1.645 * np.sqrt(lead_time * (base_d * 0.4)**2 + base_d**2 * lead_std**2)
        
        if stock_scenario == "critical":
            stock = int(np.random.uniform(0.1, 0.7) * (expected_rop * 0.4))
            on_order = int(np.random.choice([0, moq], p=[0.7, 0.3]))
        elif stock_scenario == "reorder":
            stock = int(np.random.uniform(0.7, 1.0) * expected_rop)
            on_order = 0
        elif stock_scenario == "healthy":
            stock = int(np.random.uniform(1.2, 2.2) * expected_rop)
            on_order = 0
        else:  # overstocked
            stock = int(np.random.uniform(3.0, 5.0) * expected_rop)
            on_order = 0

        skus.append({
            "sku_id": sku_code,
            "product_name": name,
            "category": cat,
            "unit_cost": float(cost),
            "selling_price": float(price),
            "supplier_lead_time_days": float(lead_time),
            "lead_time_std_days": float(lead_std),
            "moq": int(moq),
            "case_pack": int(pack),
            "current_stock": max(0, int(stock)),
            "incoming_orders": int(on_order),
        })

    df_catalog = pd.DataFrame(skus)

    # Generate daily sales history
    end_date = pd.to_datetime(end_date_str)
    date_range = pd.date_range(end=end_date, periods=history_days, freq="D")

    sales_records = []
    # Day-of-week demand multipliers: Mon(0) to Sun(6)
    # Mon: 0.85, Tue: 0.88, Wed: 0.92, Thu: 0.95, Fri: 1.20, Sat: 1.35, Sun: 1.15
    dow_multipliers = np.array([0.85, 0.88, 0.92, 0.95, 1.20, 1.35, 1.15])

    for i, sku_row in df_catalog.iterrows():
        sku_id = sku_row["sku_id"]
        base_d = base_demands[i]
        price = sku_row["selling_price"]

        # Generate daily sales using Poisson / Negative Binomial approximation
        dows = date_range.dayofweek.to_numpy()
        dow_factors = dow_multipliers[dows]

        # Monthly macro trend (slight upward or seasonal movement)
        t = np.linspace(0, 1, history_days)
        trend_drift = 1.0 + 0.15 * np.sin(2 * np.pi * t) + np.random.uniform(-0.08, 0.08) * t

        # Promotional shocks (approx 4% probability on any day of a 2x-3x surge)
        promo_mask = np.random.binomial(1, 0.04, size=history_days)
        promo_boost = 1.0 + promo_mask * np.random.uniform(1.2, 2.2, size=history_days)

        expected_daily = base_d * dow_factors * trend_drift * promo_boost
        # Sample actual integer sales with Poisson noise
        daily_units = np.random.poisson(expected_daily)
        daily_units = np.maximum(0, daily_units)

        for date_val, units in zip(date_range, daily_units):
            sales_records.append({
                "date": date_val.strftime("%Y-%m-%d"),
                "sku_id": sku_id,
                "units_sold": int(units),
                "revenue": round(float(units * price), 2),
            })

    df_sales = pd.DataFrame(sales_records)

    if save_to_disk:
        os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
        df_catalog.to_csv(CATALOG_PATH, index=False)
        df_sales.to_csv(SALES_PATH, index=False)
        print(f"[+] Dataset saved to:\n    - {CATALOG_PATH} ({len(df_catalog)} SKUs)\n    - {SALES_PATH} ({len(df_sales)} daily transaction records)")

    return df_catalog, df_sales


if __name__ == "__main__":
    generate_retail_dataset()
