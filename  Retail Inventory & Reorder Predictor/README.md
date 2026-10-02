# Retail Inventory & Reorder Predictor (NumPy & Pandas)

An advanced retail inventory forecasting and replenishment optimization engine built in Python using **NumPy** and **Pandas**.

The system automates the decision-making process for supply chain and store managers by computing dynamic safety stocks, reorder points (ROP), Economic Order Quantities (EOQ), Pareto ABC classifications, and stockout runway projections. It includes a discrete-event simulation engine to backtest and benchmark predictive replenishment policies against naive periodic reordering.

---

## Key Features

- **Vectorized Demand Velocity & Volatility Engine**: Calculates rolling mean daily sales ($\mu_d$), demand volatility ($\sigma_d$), and linear demand trend velocity using NumPy least-squares regression.
- **Dual-Variance Safety Stock Modeling**: Protects against both fluctuating customer demand and supplier lead-time delays:
  $$SS = Z \times \sqrt{\bar{L} \cdot \sigma_d^2 + \bar{d}^2 \cdot \sigma_L^2}$$
- **Dynamic Reorder Point (ROP)**:
  $$ROP = (\bar{d} \times \bar{L}) + SS$$
- **Constrained Economic Order Quantity (EOQ)**: Implements Wilson's lot-sizing formulation bounded by supplier Minimum Order Quantity (MOQ) and batch pack sizes:
  $$EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
- **Pareto ABC Classification**: Automatically stratifies SKUs into Class A (top 80% revenue), Class B (next 15%), and Class C (bottom 5%) using Pandas cumulative revenue ranking.
- **Inventory Runway & Health Diagnostics**: Classifies every SKU into `CRITICAL`, `REORDER NEEDED`, `OPTIMAL`, or `OVERSTOCKED` with calculated days of remaining runway.
- **Replenishment Simulation Engine**: 45-day Monte Carlo operational simulation measuring fill rate, stockouts, lost revenue, holding costs, and order frequencies.
- **Executive Reporting & Visual Dashboards**: Formatted terminal KPI cards, actionable purchase order CSV exports, and 4-panel visual analytics charts generated with Matplotlib.

---

## Project Structure

```
numpy & panda project/
├── data/
│   ├── inventory_catalog.csv       # Product catalog, costs, lead times, MOQ, stock
│   ├── sales_history.csv           # Historical daily sales transactions
│   └── purchase_orders.csv         # Generated PO replenishment recommendations
├── reports/
│   └── inventory_analytics.png     # 4-panel visual health & simulation dashboard
├── src/
│   ├── __init__.py
│   ├── config.py                   # Economic rates, Z-scores, and system defaults
│   ├── data_generator.py           # Realistic synthetic retail dataset generator
│   ├── analytics_engine.py         # Vectorized NumPy/Pandas calculation engine
│   ├── simulator.py                # 45-day replenishment policy comparison simulator
│   ├── visualizer.py               # Visual analytics plotting using Matplotlib
│   └── reporter.py                 # Terminal KPI cards and CSV export formatting
├── tests/
│   ├── __init__.py
│   └── test_engine.py              # Mathematical formula unit tests
├── main.py                         # CLI entrypoint supporting all subcommands
└── README.md                       # Documentation and usage guide
```

---

## Mathematical Formulation

### 1. Dynamic Safety Stock ($SS$)
Traditional formulas only consider demand variance. This system factors in **supplier delivery uncertainty** ($\sigma_L$) as well:
$$SS = \left\lceil Z \times \sqrt{L \cdot \sigma_d^2 + \mu_d^2 \cdot \sigma_L^2} \right\rceil$$

Where:
- $Z$: Normal distribution quantile for target cycle service level ($1.645$ for $95\%$, $2.326$ for $99\%$).
- $L$: Mean supplier lead time in days.
- $\sigma_L$: Standard deviation of supplier lead time in days.
- $\mu_d$: Mean daily units sold over rolling lookback window.
- $\sigma_d$: Standard deviation of daily units sold.

### 2. Reorder Point ($ROP$)
The inventory threshold triggering a purchase order:
$$ROP = \lceil \mu_d \cdot L + SS \rceil$$

### 3. Economic Order Quantity ($EOQ$) with Constraints
Balances inventory holding cost with administrative ordering cost:
$$\text{Raw } EOQ = \sqrt{\frac{2 \cdot D \cdot S}{H}}$$
- $D = \mu_d \times 365$ (Annualized demand)
- $S$: Fixed administrative and shipping cost per order (default: $\$50.00$)
- $H = \text{holding\_rate} \times \text{unit\_cost}$ (Annual cost to carry one unit in stock)

**Operational Batching**:
$$\text{Batched } EOQ = \left\lceil \frac{\text{Raw } EOQ}{\text{Case Pack}} \right\rceil \times \text{Case Pack}$$
$$EOQ = \max(\text{MOQ}, \text{Batched } EOQ)$$

### 4. Stockout Runway
Calculates how many days until current stock on hand reaches zero:
$$\text{Runway (Days)} = \frac{\text{Current Stock}}{\mu_d}$$

---

## Quick Start & Usage

### 1. Run the Complete Pipeline
Executes dataset loading/generation, analytical audit, policy simulation, and visual dashboard creation:
```bash
python3 main.py all
```

### 2. Analytical Audit & Purchase Order Export
Audit current stock levels against ROP/EOQ and export recommended purchase orders to `data/purchase_orders.csv`:
```bash
python3 main.py analyze --service-level 0.95 --holding-rate 0.20 --order-cost 50.0
```

### 3. Replenishment Policy Simulation
Compare dynamic ROP/EOQ reordering against a fixed periodic reordering schedule over a 45-day horizon:
```bash
python3 main.py simulate --sim-days 45
```

### 4. Generate Visual Dashboard
Creates a high-resolution 4-panel dashboard in `reports/inventory_analytics.png`:
```bash
python3 main.py dashboard
```

### 5. Generate Custom Datasets
Generate custom retail datasets with arbitrary numbers of SKUs and historical sales days:
```bash
python3 main.py generate --skus 50 --days 180 --seed 42
```

---

## Running Unit Tests

The test suite validates the mathematical correctness of the formulas, zero-demand edge cases, and Pareto classification:
```bash
python3 -m unittest discover tests
```

---

## Sample Simulation Output

| Metric | Predictive (ROP/EOQ) | Naive Periodic | Difference |
| :--- | :--- | :--- | :--- |
| **Service Level / Fill Rate (%)** | **96.09%** | 92.49% | **+3.6%** |
| **Stockout SKU-Days (%)** | **6.89%** | 8.39% | **-1.5%** |
| **Lost Revenue ($)** | **\$57,119.44** | \$67,950.90 | **-\$10,831.46** |
| **Average Inventory Value ($)** | **\$125,932.68** | \$230,589.15 | **-\$104,656.47** |
| **Holding Cost ($)** | **\$3,105.19** | \$5,685.76 | **-\$2,580.57** |
| **Purchase Orders Placed** | **46** | 160 | **-114 orders** |
| **Total Operational Cost ($)** | **\$19,685.05** | \$30,673.48 | **-\$10,988.43** |

---

## Technology Stack

- **Python 3.14+**
- **NumPy**: Vectorized arrays, variance calculations, linear regression trend estimation, normal and Poisson distributions.
- **Pandas**: Grouping, rolling windows, cumulative Pareto sums, pivot tables, and CSV manipulation.
- **Matplotlib**: Dark-themed multi-panel analytical dashboards and data visualization.
