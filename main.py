"""
Neutron Star Solver Flask Web Application
Run with: python app.py
Access at: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
from numba import jit, prange
import plotly.graph_objects as go
import plotly.utils
import json
import io
import os
from werkzeug.utils import secure_filename
from scipy.optimize import brentq

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Constants
G = 6.67e-8
c = 3e10
c2 = c * c
TWO_G_C2 = 2 * G / c2
FOUR_PI = 4.0 * np.pi
P_SURF = 1e20  # Surface pressure threshold (dyn/cm²)

# EOS Configuration
EOS_CONFIG = [
    ("BSK20", "bsk20.csv", 6, "#D7009A"),
    ("DD2", "dd2.csv", 4, "#16A1BD"),
    ("APR3", "apr3.csv", 1, "#800080"),
    ("ENG", "eng.csv", 1, "#000080"),
    ("TMA", "tma.csv", 3, "#B22222"),
    ("WFF2", "wff2.csv", 1, "#BE00BB"),
    ("H4", "h4.csv", 1, "#FF69B4"),
    ("MPA1", "mpa1.csv", 4, "#CD853F"),
    ("SFHOY", "sfhoy.csv", 4, "#708090"),
    ("WFF1", "wff1.csv", 1, "#8FBC8F"),
    ("SFHO", "sfho.csv", 4, "#2F4F4F"),
    ("TM1", "tm1.csv", 4, "#FFE4B5"),
    ("APRLD", "aprld.csv", 4, "#D8BFD8"),
    ("BHBLP", "bhblp.csv", 4, "#4682B4"),
    ("DD2F", "dd2f.csv", 1, "#EE82EE"),
    ("SLY", "sly.csv", 1, "#6FAC52"),
    ("FPS", "fps.csv", 1, "#808000"),
    ("ALF2", "alf2.csv", 1, "#7FFF00")
]

# Load EOS data
eos_cache = {}

def load_eos_data():
    """Load all predefined EOS data"""
    columns = ["a", "b", "rho", "P"]
    loaded_count = 0
    
    # Check if eos directory exists
    if not os.path.exists('./eos'):
        print("ERROR: './eos' directory not found!")
        print("Please create an 'eos' folder in the same directory as this script")
        return
    
    print("\nLoading EOS files from: ./eos/")
    
    for name, filename, sampling, color in EOS_CONFIG:
        filepath = f"./eos/{filename}"
        try:
            if not os.path.exists(filepath):
                print(f"  ✗ {filename} - FILE NOT FOUND")
                continue
                
            df = pd.read_csv(filepath, usecols=columns)
            if sampling > 1:
                df = df[1::sampling]
            
            eos_cache[name] = {
                'rho': df.rho.values.astype(np.float64),
                'P': df.P.values.astype(np.float64),
                'color': color
            }
            loaded_count += 1
            print(f"  ✓ {filename} - Loaded successfully ({len(df)} points)")
        except Exception as e:
            print(f"  ✗ {filename} - Error: {e}")
    
    print(f"\nTotal EOS loaded: {loaded_count}/{len(EOS_CONFIG)}")
    
    if loaded_count == 0:
        print("\nWARNING: No EOS files were loaded!")
        print("The app will run but computations will fail.")
        print("Please ensure your EOS CSV files are in the './eos/' directory.")

# Numba-compiled functions
@jit(nopython=True, fastmath=True)
def interp_P(rho_val, rho_arr, P_arr):
    """Fast linear interpolation for P(rho)"""
    if rho_val <= rho_arr[0]:
        return P_arr[0]
    if rho_val >= rho_arr[-1]:
        return P_arr[-1]
    
    left, right = 0, len(rho_arr) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if rho_arr[mid] <= rho_val:
            left = mid
        else:
            right = mid
    
    t = (rho_val - rho_arr[left]) / (rho_arr[right] - rho_arr[left])
    return P_arr[left] + t * (P_arr[right] - P_arr[left])

@jit(nopython=True, fastmath=True)
def interp_rho(P_val, rho_arr, P_arr):
    """Fast linear interpolation for rho(P)"""
    if P_val <= P_arr[0]:
        return rho_arr[0]
    if P_val >= P_arr[-1]:
        return rho_arr[-1]
    
    left, right = 0, len(P_arr) - 1
    while right - left > 1:
        mid = (left + right) // 2
        if P_arr[mid] <= P_val:
            left = mid
        else:
            right = mid
    
    t = (P_val - P_arr[left]) / (P_arr[right] - P_arr[left])
    return rho_arr[left] + t * (rho_arr[right] - rho_arr[left])

@jit(nopython=True, fastmath=True, inline='always')
def compute_derivatives(r, m, P, rho):
    """Compute TOV equation derivatives"""
    dr_dr = 1.0
    dm_dr = FOUR_PI * r * r * rho
    
    numerator = -G * m * rho / (r * r)
    factor1 = 1.0 + P / (rho * c2)
    factor2 = 1.0 + FOUR_PI * P * r * r * r / (m * c2)
    denominator = 1.0 - TWO_G_C2 * m / r
    
    dP_dr = (numerator * factor1 * factor2) / denominator
    
    return dr_dr, dm_dr, dP_dr

@jit(nopython=True, fastmath=True)
def ns_solve_numba(rho_0, rho_arr, P_arr, dr=500.0, max_iter=80000, tol=1e-10):
    """Ultra-fast RK4 solver with Numba JIT compilation"""
    P_0 = interp_P(rho_0, rho_arr, P_arr)
    
    r = 500.0
    m = 1.0
    P = P_0
    
    for i in range(max_iter):
        rho1 = interp_rho(P, rho_arr, P_arr)
        k1_r, k1_m, k1_P = compute_derivatives(r, m, P, rho1)
        
        r2 = r + 0.5 * dr * k1_r
        m2 = m + 0.5 * dr * k1_m
        P2 = P + 0.5 * dr * k1_P
        rho2 = interp_rho(P2, rho_arr, P_arr)
        k2_r, k2_m, k2_P = compute_derivatives(r2, m2, P2, rho2)
        
        r3 = r + 0.5 * dr * k2_r
        m3 = m + 0.5 * dr * k2_m
        P3 = P + 0.5 * dr * k2_P
        rho3 = interp_rho(P3, rho_arr, P_arr)
        k3_r, k3_m, k3_P = compute_derivatives(r3, m3, P3, rho3)
        
        r4 = r + dr * k3_r
        m4 = m + dr * k3_m
        P4 = P + dr * k3_P
        rho4 = interp_rho(P4, rho_arr, P_arr)
        k4_r, k4_m, k4_P = compute_derivatives(r4, m4, P4, rho4)
        
        r += dr * (k1_r + 2*k2_r + 2*k3_r + k4_r) / 6.0
        m += dr * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6.0
        P += dr * (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6.0
        
        if P / P_0 < tol or P < 0:
            break
    
    return r, m, P

@jit(nopython=True, parallel=True, fastmath=True)
def solve_all_densities(rho_array, rho_eos, P_eos):
    """Solve for all densities in parallel"""
    n = len(rho_array)
    results = np.zeros((3, n))
    
    for i in prange(n):
        r, m, P = ns_solve_numba(rho_array[i], rho_eos, P_eos)
        results[0, i] = r
        results[1, i] = m
        results[2, i] = P
    
    return results

def compute_eos(eos_name, rho_arr, P_arr):
    """Compute neutron star properties for given EOS"""
    # Generate density array
    rho = np.concatenate([
        np.arange(2.5e14, 1e15, 0.5e13),
        np.arange(1e15, 4e15, 0.5e14)
    ])
    
    # Solve
    results = solve_all_densities(rho, rho_arr, P_arr)
    
    R = results[0, :] / 1e5  # Convert to km
    M = results[1, :] / 2e33  # Convert to solar masses
    
    return {
        'name': eos_name,
        'rho': rho.tolist(),
        'R': R.tolist(),
        'M': M.tolist(),
        'max_mass': float(np.max(M)),
        'max_radius': float(R[np.argmax(M)]),
        'raw_rho': rho_arr.tolist(),  # Store raw EOS data
        'raw_P': P_arr.tolist()       # Store raw EOS data
    }

def compute_inertia(m, r):
    """Compute moment of inertia with robust error handling"""
    try:
        # Ensure inputs are numpy arrays
        m = np.asarray(m, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        
        # Filter out invalid values before computation
        valid_mask = (m > 0) & (r > 0) & np.isfinite(m) & np.isfinite(r)
        
        if not np.any(valid_mask):
            print("Warning: No valid M,R pairs for inertia computation")
            return np.array([])
        
        # Apply mask
        m_valid = m[valid_mask]
        r_valid = r[valid_mask]
        
        # Compute compactness (dimensionless)
        # M is in solar masses, R is in km
        # Convert to geometric units: M_sun = 1.477 km in geometric units
        m_km = m_valid * 1.477  # Convert solar masses to km (geometric units)
        compactness = m_km / r_valid
        
        # Avoid extreme compactness values that cause numerical issues
        compactness = np.clip(compactness, 0, 0.5)  # Max compactness ~0.5 for neutron stars
        
        # Compute moment of inertia
        # I = k * M * R^2 where k depends on compactness
        k = 0.237 + 0.008  # Base value
        k = k * (1 + 4.2 * compactness + 90 * compactness**4)
        
        # I in units of 10^45 g·cm²
        # M in M_sun (1.989e33 g), R in km (1e5 cm)
        I = k * m_valid * (r_valid * 1e5)**2 * 1.989e33 / 1e45
        
        # Final safety check
        I = np.where(np.isfinite(I) & (I > 0), I, np.nan)
        
        # Create full result array with NaN for invalid entries
        result = np.full_like(m, np.nan, dtype=np.float64)
        result[valid_mask] = I
        
        return result
        
    except Exception as e:
        print(f"Error in inertia computation: {e}")
        import traceback
        traceback.print_exc()
        return np.full_like(m, np.nan) if hasattr(m, '__len__') else np.nan

@jit(nopython=True, fastmath=True)
def tov_derivatives_profile(r, m, P, rho):
    """
    Compute TOV equation derivatives - CORRECTED VERSION
    
    dm/dr = 4π r² ρ (mass density in g/cm³)
    dP/dr = -G(ε + P/c²)(m + 4πr³P/c²) / [r(r - 2Gm/c²)]
    
    where ε = ρc² is the energy density
    
    Parameters:
    - r: radius (cm)
    - m: enclosed mass (g)
    - P: pressure (dyn/cm²)
    - rho: mass density (g/cm³)
    
    Returns: (dm/dr, dP/dr)
    """
    if r < 1e-10:
        return 0.0, 0.0
    
    # Constants (from global scope)
    G = 6.67e-8
    c = 3e10
    c2 = c * c
    FOUR_PI = 4.0 * np.pi
    TWO_G_C2 = 2 * G / c2
    
    # dm/dr = 4π r² ρ (mass density!)
    dm_dr = FOUR_PI * r * r * rho
    
    # For pressure equation, use energy density ε = ρc²
    epsilon = rho * c2
    
    # dP/dr equation
    numerator = -G * (epsilon + P / c2) * (m + FOUR_PI * r * r * r * P / c2)
    denominator = r * (r - TWO_G_C2 * m)
    
    if abs(denominator) < 1e-30:
        return dm_dr, 0.0
    
    dP_dr = numerator / denominator
    
    return dm_dr, dP_dr

@jit(nopython=True, fastmath=True)
def integrate_tov_profile(rho_c, rho_arr, P_arr, r0=500.0, dr=500.0, max_iter=80000):
    """
    Integrate TOV equations - CORRECTED VERSION
    
    Parameters:
    - rho_c: Central density (g/cm³)
    - rho_arr, P_arr: EOS arrays
    - r0: Starting radius (cm), default 500 cm = 0.005 km
    - dr: Step size (cm)
    - max_iter: Maximum iterations
    
    Returns:
    - r_array, m_array, P_array, rho_array, R_final, M_final
    """
    # Constants
    G = 6.67e-8
    c = 3e10
    c2 = c * c
    FOUR_PI = 4.0 * np.pi
    TWO_G_C2 = 2 * G / c2
    P_SURF = 1e20
    
    # Get central pressure from EOS
    P_c = interp_P(rho_c, rho_arr, P_arr)
    
    # Initial conditions at r0
    r = r0
    # CORRECTED: m(r0) = (4/3)π r0³ ρc (NOT ρc c²)
    m = (4.0/3.0) * np.pi * r0**3 * rho_c
    P = P_c
    
    # Storage arrays
    r_arr_out = np.zeros(max_iter)
    m_arr_out = np.zeros(max_iter)
    P_arr_out = np.zeros(max_iter)
    rho_arr_out = np.zeros(max_iter)
    
    # Store initial values
    r_arr_out[0] = r
    m_arr_out[0] = m
    P_arr_out[0] = P
    rho_arr_out[0] = rho_c
    
    idx = 1
    
    # RK4 integration
    for i in range(1, max_iter):
        rho = interp_rho(P, rho_arr, P_arr)
        
        # RK4 step
        k1_m, k1_P = tov_derivatives_profile_v2(r, m, P, rho)
        
        r2 = r + 0.5 * dr
        m2 = m + 0.5 * dr * k1_m
        P2 = P + 0.5 * dr * k1_P
        rho2 = interp_rho(P2, rho_arr, P_arr)
        k2_m, k2_P = tov_derivatives_profile_v2(r2, m2, P2, rho2)
        
        r3 = r + 0.5 * dr
        m3 = m + 0.5 * dr * k2_m
        P3 = P + 0.5 * dr * k2_P
        rho3 = interp_rho(P3, rho_arr, P_arr)
        k3_m, k3_P = tov_derivatives_profile_v2(r3, m3, P3, rho3)
        
        r4 = r + dr
        m4 = m + dr * k3_m
        P4 = P + dr * k3_P
        rho4 = interp_rho(P4, rho_arr, P_arr)
        k4_m, k4_P = tov_derivatives_profile_v2(r4, m4, P4, rho4)
        
        # Update
        r += dr
        m += dr * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6.0
        P += dr * (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6.0
        
        # Stop if pressure drops below surface threshold
        if P <= P_SURF or P < 0:
            break
        
        # Store values
        rho_arr_out[idx] = interp_rho(P, rho_arr, P_arr)
        r_arr_out[idx] = r
        m_arr_out[idx] = m
        P_arr_out[idx] = P
        idx += 1
    
    # Return only filled portion
    return (r_arr_out[:idx], m_arr_out[:idx], P_arr_out[:idx], 
            rho_arr_out[:idx], r, m)

def find_rho_c_for_mass_v2(target_mass_solar, rho_arr, P_arr, M_rho_grid=None):
    """
    Find central density for target mass - CORRECTED VERSION
    """
    target_mass_g = target_mass_solar * 2e33
    
    if M_rho_grid is not None:
        rho_grid, M_grid = M_rho_grid
        max_idx = np.argmax(M_grid)
        M_max = M_grid[max_idx]
        
        print(f"  M_max for this EOS: {M_max:.3f} M☉")
        print(f"  Target mass: {target_mass_solar:.3f} M☉")
        
        if target_mass_solar > M_max:
            raise ValueError(f"Target mass {target_mass_solar:.3f} M☉ exceeds maximum "
                           f"{M_max:.3f} M☉ for this EOS")
        
        M_min = M_grid[0]
        if target_mass_solar < M_min:
            raise ValueError(f"Target mass {target_mass_solar:.3f} M☉ is below minimum "
                           f"{M_min:.3f} M☉ for this EOS")
        
        stable_rho = rho_grid[:max_idx+1]
        stable_M = M_grid[:max_idx+1]
        
        idx = np.searchsorted(stable_M, target_mass_solar)
        
        if idx == 0:
            rho_low = stable_rho[0]
            rho_high = stable_rho[1]
        elif idx >= len(stable_M):
            rho_low = stable_rho[-2]
            rho_high = stable_rho[-1]
        else:
            rho_low = stable_rho[idx-1]
            rho_high = stable_rho[idx]
        
        print(f"  Bracketing: rho_c ∈ [{rho_low:.3e}, {rho_high:.3e}] g/cm³")
    else:
        rho_low = 1e14
        rho_high = 5e15
    
    def mass_difference(rho_c):
        try:
            _, _, _, _, _, M = integrate_tov_profile_v2(rho_c, rho_arr, P_arr)
            diff = M - target_mass_g
            return diff
        except Exception as e:
            print(f"    Error at rho_c={rho_c:.3e}: {e}")
            return np.inf
    
    f_low = mass_difference(rho_low)
    f_high = mass_difference(rho_high)
    
    print(f"  f(rho_low) = {f_low/2e33:.6f} M☉")
    print(f"  f(rho_high) = {f_high/2e33:.6f} M☉")
    
    if f_low * f_high > 0:
        print(f"  Initial bracket has same sign, expanding search...")
        
        if abs(f_low) < abs(f_high):
            rho_low = max(rho_low * 0.5, rho_arr[0])
            f_low = mass_difference(rho_low)
            print(f"  Expanded low: rho={rho_low:.3e}, f={f_low/2e33:.6f} M☉")
        else:
            rho_high = min(rho_high * 1.5, rho_arr[-1])
            f_high = mass_difference(rho_high)
            print(f"  Expanded high: rho={rho_high:.3e}, f={f_high/2e33:.6f} M☉")
        
        if f_low * f_high > 0:
            raise ValueError(
                f"Cannot bracket target mass {target_mass_solar:.3f} M☉. "
                f"Mass at rho_low={rho_low:.2e}: {(f_low/2e33 + target_mass_solar):.3f} M☉, "
                f"Mass at rho_high={rho_high:.2e}: {(f_high/2e33 + target_mass_solar):.3f} M☉"
            )
    
    print(f"  Starting Brent's method...")
    rho_c = brentq(mass_difference, rho_low, rho_high, xtol=1e12, maxiter=100)
    print(f"  Converged to rho_c = {rho_c:.3e} g/cm³")
    
    return rho_c
    
def create_plot(results, plot_type):
    """Create Plotly figure based on plot type with proper error handling"""
    try:
        fig = go.Figure()
        
        if not results:
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        if plot_type == 'mass-rho':
            for result in results:
                if 'rho' in result and 'M' in result and len(result['rho']) > 0 and len(result['M']) > 0:
                    fig.add_trace(go.Scatter(
                        x=result['rho'],
                        y=result['M'],
                        mode='lines',
                        name=result['name'],
                        line=dict(color=result.get('color', '#000000'), width=2)
                    ))
            fig.update_layout(
                title='Mass vs Central Density',
                xaxis_title='ρc (g/cm³)',
                yaxis_title='Mass (M☉)',
                xaxis_type='log',
                template='plotly_dark'
            )
        
        elif plot_type == 'radius-rho':
            for result in results:
                if 'rho' in result and 'R' in result and len(result['rho']) > 0 and len(result['R']) > 0:
                    fig.add_trace(go.Scatter(
                        x=result['rho'],
                        y=result['R'],
                        mode='lines',
                        name=result['name'],
                        line=dict(color=result.get('color', '#000000'), width=2)
                    ))
            fig.update_layout(
                title='Radius vs Central Density',
                xaxis_title='ρc (g/cm³)',
                yaxis_title='Radius (km)',
                xaxis_type='log',
                yaxis=dict(range=[9, 30]),
                template='plotly_dark'
            )
        
        elif plot_type == 'mass-radius':
            for result in results:
                if 'R' in result and 'M' in result and len(result['R']) > 0 and len(result['M']) > 0:
                    fig.add_trace(go.Scatter(
                        x=result['R'],
                        y=result['M'],
                        mode='lines',
                        name=result['name'],
                        line=dict(color=result.get('color', '#000000'), width=2)
                    ))
            fig.update_layout(
                title='Mass vs Radius',
                xaxis_title='Radius (km)',
                yaxis_title='Mass (M☉)',
                xaxis=dict(range=[8, 20]),
                yaxis=dict(range=[0, 3]),
                template='plotly_dark'
            )
        
        elif plot_type == 'inertia':
            traces_added = 0
            for result in results:
                try:
                    if 'M' not in result or 'R' not in result:
                        print(f"Skipping {result.get('name', 'unknown')}: Missing M or R data")
                        continue
                    
                    M = np.array(result['M'])
                    R = np.array(result['R'])
                    
                    if len(M) == 0 or len(R) == 0:
                        print(f"Skipping {result.get('name', 'unknown')}: Empty arrays")
                        continue
                    
                    # Find stable branch (up to maximum mass)
                    max_idx = np.argmax(M)
                    if max_idx < 2:  # Need at least a few points
                        print(f"Skipping {result.get('name', 'unknown')}: Insufficient stable points")
                        continue
                    
                    M_stable = M[:max_idx+1]
                    R_stable = R[:max_idx+1]
                    
                    print(f"Computing inertia for {result.get('name', 'unknown')}: {len(M_stable)} points")
                    print(f"  M range: [{M_stable.min():.3f}, {M_stable.max():.3f}]")
                    print(f"  R range: [{R_stable.min():.3f}, {R_stable.max():.3f}]")
                    
                    # Compute inertia
                    I = compute_inertia(M_stable, R_stable)
                    
                    # Filter valid values (not NaN, not Inf, positive)
                    valid_mask = np.isfinite(I) & (I > 0) & (I < 1000)  # Sanity check on upper bound
                    
                    valid_count = np.sum(valid_mask)
                    print(f"  Valid inertia points: {valid_count}/{len(I)}")
                    
                    if valid_count > 0:
                        M_plot = M_stable[valid_mask]
                        I_plot = I[valid_mask]
                        
                        print(f"  I range: [{I_plot.min():.2f}, {I_plot.max():.2f}]")
                        
                        fig.add_trace(go.Scatter(
                            x=M_plot.tolist(),
                            y=I_plot.tolist(),
                            mode='lines',
                            name=result['name'],
                            line=dict(color=result.get('color', '#000000'), width=2)
                        ))
                        traces_added += 1
                    else:
                        print(f"  No valid inertia values for {result.get('name', 'unknown')}")
                    
                except Exception as e:
                    print(f"Error computing inertia for {result.get('name', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"Total traces added to inertia plot: {traces_added}")
            
            fig.update_layout(
                title='Moment of Inertia',
                xaxis_title='Mass (M☉)',
                yaxis_title='I (10⁴⁵ g·cm²)',
                xaxis=dict(range=[1, 2.7]),
                template='plotly_dark'
            )
        
        elif plot_type == 'eos':
            # NEW: Plot raw EoS data with logarithmic axes
            for result in results:
                if 'raw_rho' in result and 'raw_P' in result and len(result['raw_rho']) > 0 and len(result['raw_P']) > 0:
                    fig.add_trace(go.Scatter(
                        x=result['raw_rho'],
                        y=result['raw_P'],
                        mode='lines',
                        name=result['name'],
                        line=dict(color=result.get('color', '#000000'), width=2)
                    ))
            fig.update_layout(
                title='Equation of State: Pressure vs Density',
                xaxis_title='ρ (g/cm³)',
                yaxis_title='P (dyn/cm²)',
                xaxis_type='log',
                yaxis_type='log',
                template='plotly_dark'
            )
        
        else:
            fig.add_annotation(text=f"Invalid plot type: {plot_type}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        
        # Ensure we have at least one trace
        if len(fig.data) == 0:
            fig.add_annotation(text="No valid data to display", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            print("WARNING: No traces added to plot!")
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    except Exception as e:
        print(f"Error in create_plot: {e}")
        import traceback
        traceback.print_exc()
        # Return a basic error plot
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating plot: {str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def generate_csv_data(results, plot_type):
    """Generate CSV data for the selected plot type"""
    if not results:
        return "No data available"
    
    csv_lines = []
    
    if plot_type == 'mass-rho':
        csv_lines.append("EOS,Central Density (g/cm³),Mass (M☉)")
        for result in results:
            if 'rho' in result and 'M' in result:
                for rho, mass in zip(result['rho'], result['M']):
                    csv_lines.append(f"{result['name']},{rho},{mass}")
    
    elif plot_type == 'radius-rho':
        csv_lines.append("EOS,Central Density (g/cm³),Radius (km)")
        for result in results:
            if 'rho' in result and 'R' in result:
                for rho, radius in zip(result['rho'], result['R']):
                    csv_lines.append(f"{result['name']},{rho},{radius}")
    
    elif plot_type == 'mass-radius':
        csv_lines.append("EOS,Radius (km),Mass (M☉)")
        for result in results:
            if 'R' in result and 'M' in result:
                for radius, mass in zip(result['R'], result['M']):
                    csv_lines.append(f"{result['name']},{radius},{mass}")
    
    elif plot_type == 'inertia':
        csv_lines.append("EOS,Mass (M☉),Inertia (10⁴⁵ g·cm²)")
        for result in results:
            if 'M' in result and 'R' in result:
                M = np.array(result['M'])
                R = np.array(result['R'])
                
                # Find stable branch
                max_idx = np.argmax(M)
                if max_idx >= 2:
                    M_stable = M[:max_idx+1]
                    R_stable = R[:max_idx+1]
                    I = compute_inertia(M_stable, R_stable)
                    
                    for mass, inertia in zip(M_stable, I):
                        if np.isfinite(inertia) and inertia > 0:
                            csv_lines.append(f"{result['name']},{mass},{inertia}")
    
    elif plot_type == 'eos':
        csv_lines.append("EOS,Density (g/cm³),Pressure (dyn/cm²)")
        for result in results:
            if 'raw_rho' in result and 'raw_P' in result:
                for rho, pressure in zip(result['raw_rho'], result['raw_P']):
                    csv_lines.append(f"{result['name']},{rho},{pressure}")
    
    else:
        return f"Unknown plot type: {plot_type}"
    
    return "\n".join(csv_lines)

# Routes
@app.route('/')
def index():
    """Render main page"""
    # Only include EOS that are actually loaded
    eos_list = [{'name': name, 'color': color} 
                for name, _, _, color in EOS_CONFIG 
                if name in eos_cache]
    
    print(f"\nServing index page with {len(eos_list)} available EOS")
    return render_template('index.html', eos_list=eos_list)

@app.route('/compute', methods=['POST'])
def compute():
    """Compute neutron star properties"""
    try:
        data = request.get_json()
        selected_eos = data.get('eos', [])
        plot_type = data.get('plot_type', 'mass-radius')
        
        print(f"\nCompute request received:")
        print(f"  Selected EOS: {selected_eos}")
        print(f"  Plot type: {plot_type}")
        
        if not selected_eos:
            return jsonify({'error': 'No EOS selected'}), 400
        
        results = []
        
        for eos_name in selected_eos:
            if eos_name not in eos_cache:
                print(f"  Warning: {eos_name} not found in cache")
                continue
                
            print(f"  Computing {eos_name}...")
            result = compute_eos(
                eos_name,
                eos_cache[eos_name]['rho'],
                eos_cache[eos_name]['P']
            )
            result['color'] = eos_cache[eos_name]['color']
            results.append(result)
            print(f"    ✓ Max mass: {result['max_mass']:.3f} M☉, Max radius: {result['max_radius']:.3f} km")
        
        plot_json = create_plot(results, plot_type)
        
        return jsonify({
            'plot': plot_json,
            'results': results
        })
        
    except Exception as e:
        print(f"Error in compute: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload custom EOS file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Check for required columns
        if 'rho' not in df.columns or 'P' not in df.columns:
            return jsonify({'error': 'CSV must have "rho" and "P" columns'}), 400
        
        # Store in cache
        eos_name = secure_filename(file.filename).replace('.csv', '').upper()
        eos_cache[eos_name] = {
            'rho': df['rho'].values.astype(np.float64),
            'P': df['P'].values.astype(np.float64),
            'color': '#FF0000'  # Custom EOS in red
        }
        
        return jsonify({
            'success': True,
            'name': eos_name,
            'color': '#FF0000'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download CSV with the calculated data of the selected plot type"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        plot_type = data.get('plot_type', 'mass-radius')
        
        if not results:
            return jsonify({'error': 'No data available for download'}), 400
        
        csv_data = generate_csv_data(results, plot_type)
        
        # Create in-memory file
        output = io.StringIO()
        output.write(csv_data)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'neutron_star_{plot_type}_data.csv'
        )
        
    except Exception as e:
        print(f"Error in download_csv: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Initialize the application
if __name__ == '__main__':
    print("Starting Neutron Star Solver...")
    load_eos_data()
    print("\nApplication ready! Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)