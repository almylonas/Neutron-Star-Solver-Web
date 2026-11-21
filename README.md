# Neutron Star Solver

A Flask-based web application for solving **Tolman–Oppenheimer–Volkoff (TOV)** equations for neutron stars using various **Equations of State (EOS)**.

This tool computes and visualizes relationships between **mass**, **radius**, **central density**, and **moment of inertia** for neutron stars.  
It supports both **built-in EOS datasets** and **custom user-uploaded EOS files**.

---

## Features

- Numerical integration of the TOV equations with an adaptive Dormand–Prince 5(4) integrator compiled with Numba (results cached per EOS).  
- Interactive Plotly visualizations:
  - Mass–Radius (M–R) curves  
  - Radius vs. Central Density  
  - Mass vs. Central Density  
  - Moment of Inertia  
  - Density profile ρ(r) of a single star, chosen by mass (M☉) or central density (g/cm³)  
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

4. **Production / Docker**
    ```bash
    gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 main:app
    # or
    docker build -t ns-solver . && docker run -p 5000:5000 ns-solver
    ```
    Set `FLASK_DEBUG=1` to run `python main.py` with the Flask debugger.

5. **Vercel**
    No configuration needed: Vercel detects the Flask `app` in `main.py`, installs
    `requirements.txt` and uses Python 3.12 (from `.python-version`). Push the repo or run
    `vercel deploy`. The code folder is read-only on Vercel, so the Numba compile cache is
    written to `/tmp`; the first request on a fresh instance takes a few seconds while the
    solver compiles.

---

## Density profile ρ(r)

Choose **Density Profile ρ(r)**, then either

- **Mass (M☉)**: for every selected EOS, the central density ρc giving that mass on the
  stable branch (ρc below the maximum-mass configuration) is found by root-finding, or
- **Central density ρc (g/cm³)**: the same ρc is used for every selected EOS.

The plot shows ρ(r) from the centre to the surface (P = 10⁻¹⁰ P_c), with an optional
logarithmic axis to resolve the crust, and a reference line at nuclear saturation density.
The CSV export contains r (km), ρ (g/cm³), enclosed mass m(r) (M☉) and P (dyn/cm²).

