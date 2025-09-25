# Optimal EV Battery Swapping Station Placement using CLIMFO

This project is a Python implementation of the CLIMFO (Chaos Levy-flight Moth-flame Optimization) algorithm for solving the sizing and locating planning problem of Electric Vehicle Battery Swapping Stations (EVBSS). The methodology is based on the academic paper:

**"Sizing and Locating Planning of EV Battery Swapping Stations Considering Centralized Charging Station of Battery"** by Zixiang Dai, et al. (2025 2nd International Symposium on New Energy Technologies and Power Systems - NETPS).

## Project Goal

The objective is to create a robust and well-documented tool that determines the optimal number and locations of EVBSS to minimize total annual costs, including investment, user time, and operational expenses.

## Core Components
* **CLIMFO Algorithm:** The main optimization engine, combining Moth-Flame Optimization (MFO) with Chaos Mapping and a Levy-flight strategy.
* **Cost Function Module:** A detailed model to calculate the total annual cost of a given network configuration.
* **Voronoi Diagram:** Used for spatially assigning demand points to their nearest service station.

## Getting Started

### Prerequisites
* Python 3.8+
* Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)rodolfodomar/ev-swapping-station-optimizer-climfo.git
    cd ev-swapping-station-optimizer-climfo
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate on macOS/Linux
    source .venv/bin/activate

    # Activate on Windows
    # .\.venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.