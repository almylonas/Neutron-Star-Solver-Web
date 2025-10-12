# Neutron Star Solver

A Flask-based web application for solving **Tolman–Oppenheimer–Volkoff (TOV)** equations for neutron stars using various **Equations of State (EOS)**.

This tool computes and visualizes relationships between **mass**, **radius**, **central density**, and **moment of inertia** for neutron stars.  
It supports both **built-in EOS datasets** and **custom user-uploaded EOS files**.

---

## Features

- Numerical integration of the TOV equations using Numba-accelerated solvers.  
- Interactive Plotly visualizations:
  - Mass–Radius (M–R) curves  
  - Radius vs. Central Density  
  - Mass vs. Central Density  
  - Moment of Inertia  
- Upload custom EOS `.csv` files with `rho` and `P` columns.
- Export results as `.csv`.
- Dark mode UI with TailwindCSS.
- Supports 18 built-in EOS models.

---

## Built-in EOS Files

Place EOS files in the `./eos/` directory before running the app.  
Each CSV file must include at least these columns:

| Column | Description           |
|:--------|:----------------------|
| `rho`   | Mass density (g/cm³)  |
| `P`     | Pressure (dyn/cm²)    |


---

## Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/neutron-star-solver.git
   cd neutron-star-solver
   ```
2. **Install dependenced**
    ```bash
    pip install -r requirements.txt
    ```
3. **Running the web app**
    ```bash
   python main.py
   ```