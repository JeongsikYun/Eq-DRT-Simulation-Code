# Eq-DRT Simulation Framework

This repository contains the simulation source code and generated datasets for the research paper:

**"Equity-Aware Routing of Demand-Responsive Transit: Multi-objective Optimization Using Conditional Value at Risk"**

Submitted to *Sustainable Cities and Society*.

## 1. Overview

This study proposes an **Equity-aware DRT (Eq-DRT)** routing framework that minimizes the Conditional Value-at-Risk (CVaR) of passenger delays. The simulation is based on a **Dynamic Dial-a-Ride Problem (DDARP)** formulation solved using a hybrid Adaptive Large Neighborhood Search (ALNS) algorithm. The environment is constructed using real-world population grid data from Daejeon, South Korea.

### Objective Function
The core objective function used throughout the simulation and analysis is:

$$
\mathrm{OFV} = (1-\alpha)\,\mathrm{Mean\_Delay} + \alpha\,\mathrm{Tail\_Delay}\,(\mathrm{CVaR}_{30\%})
$$

* **$\alpha = 0.0$**: Mean delay only (Efficiency-focused).
* **$\alpha = 1.0$**: Tail delay only (Equity/Risk-focused).
* **$0 < \alpha < 1$**: Balance between efficiency and equity.

---

## 2. Repository Structure & Scripts

The repository consists of four main Python scripts. Each script is designed to perform a specific part of the simulation pipeline or to reproduce specific figures in the manuscript.

### `darp_simulation_engine.py`
**Purpose:** The main simulation engine for the Dial-a-Ride Problem with Adaptive Large Neighborhood Search (ALNS).

* **Functionality:**
    * Runs large-scale DARP simulations over a grid of **alpha** and **demand** levels.
    * Uses ALNS with multiple destroy/repair operators and Simulated Annealing for re-optimization.
    * Supports delta evaluation and parallel processing via `ProcessPoolExecutor`.
* **Outputs:** A sweep directory containing summary CSVs (`1_all_simulations_summary.csv`) and detailed logs (`detailed_results/`).
* **Note:** This script generates the raw data found in the `Alpha_Demand_Sweep_Results/` directory.

### `analyze_system_performance.py`
**Purpose:** Performs macro-level analysis of the system and generates tradeoff visualizations.

* **Reproduces Manuscript Figures:**
    * **Figure 3:** Gini coefficient of delay statistics across demand rates.
    * **Figure 4:** Trade-off between Average Delay and Gini Coefficient (Impact of parameter $\alpha$).
    * **Figure 7:** Pareto frontiers of system efficiency versus equity.
* **Functionality:**
    * Reads a sweep directory and offers an interactive menu.
    * Generates Combined 2×2 tradeoff grids, effectiveness analysis, and demand-specific statistics.

### `analyze_alpha_detail.py`
**Purpose:** Performs micro-level statistical analysis comparing a target $\alpha$ against the baseline ($\alpha=0.0$).

* **Reproduces Manuscript Figures:**
    * **Figure 5:** Delay distribution comparison (histograms) for different $\alpha$ values ($0.0, 0.1, 0.5$).
* **Functionality:**
    * Analyzes time-series data and detailed logs.
    * Produces OFV trajectories, CVaR ratios, Lorenz curves, and CDFs of delay.

### `visualize_case.py`
**Purpose:** Visualizes specific service timelines and case distributions.

* **Reproduces Manuscript Figures:**
    * **Figure 6:** Passenger service timeline (Gantt chart) for a representative simulation instance.
* **Functionality:**
    * **Mode 1:** Case distribution (Win-Win / Trade-Off / Lose-Lose) analysis.
    * **Mode 2:** Alpha scatter plots.
    * **Mode 4:** Passenger service Gantt charts (Pickup/Drop-off visualization).

---

## 3. Data Directories

* **`data/`**: Contains the input files required to initialize the simulation environment (e.g., Daejeon population grid shapefiles).
* **`Alpha_Demand_Sweep_Results/`**: Contains the **pre-generated simulation results** used in the manuscript.
    * **Note to Reviewers:** Since the full simulation sweep is computationally intensive, we provide the complete output dataset in this directory. You can run the analysis scripts immediately using this data to reproduce the figures without re-running the simulation.

---

## 4. Installation & Requirements

To run the scripts, please install the required Python dependencies:

```bash
pip install -r requirements.txt
