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
    """
    Enhanced version of ns_solve_numba that returns full profile arrays
    Uses the SAME integration logic as your working ns_solve_numba
    """
    P_0 = interp_P(rho_0, rho_arr, P_arr)
    
    r = 500.0
    m = 1.0
    P = P_0
    
    # Storage for profile
    r_profile = np.zeros(max_iter)
    m_profile = np.zeros(max_iter)
    P_profile = np.zeros(max_iter)
    rho_profile = np.zeros(max_iter)
    
    # Store initial values
    r_profile[0] = r
    m_profile[0] = m
    P_profile[0] = P
    rho_profile[0] = rho_0
    
    idx = 1
    
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
        
        # Store profile data
        if idx < max_iter:
            r_profile[idx] = r
            m_profile[idx] = m
            P_profile[idx] = P
            rho_profile[idx] = interp_rho(P, rho_arr, P_arr)
            idx += 1
    
    # Return profile arrays (trimmed) plus final R and M
    return (r_profile[:idx], m_profile[:idx], P_profile[:idx], 
            rho_profile[:idx], r, m)

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
        'max_radius': float(R[np.argmax(M)])
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
    
    # Check if central density is within EOS range
    if rho_c < rho_arr[0] or rho_c > rho_arr[-1]:
        # Return invalid result
        r_arr_out = np.zeros(1)
        m_arr_out = np.array([np.inf])
        P_arr_out = np.zeros(1)
        rho_arr_out = np.zeros(1)
        return (r_arr_out, m_arr_out, P_arr_out, rho_arr_out, r0, np.inf)
    
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
        # Check if pressure is still in valid range
        if P < P_arr[0] or P > P_arr[-1]:
            break
            
        rho = interp_rho(P, rho_arr, P_arr)
        
        # Safety check
        if not np.isfinite(rho) or not np.isfinite(m) or not np.isfinite(P):
            break
        
        # RK4 step
        k1_m, k1_P = tov_derivatives_profile(r, m, P, rho)
        
        r2 = r + 0.5 * dr
        m2 = m + 0.5 * dr * k1_m
        P2 = P + 0.5 * dr * k1_P
        
        # Check if P2 is valid
        if P2 < P_arr[0]:
            P2 = P_arr[0]
        if P2 > P_arr[-1]:
            P2 = P_arr[-1]
            
        rho2 = interp_rho(P2, rho_arr, P_arr)
        k2_m, k2_P = tov_derivatives_profile(r2, m2, P2, rho2)
        
        r3 = r + 0.5 * dr
        m3 = m + 0.5 * dr * k2_m
        P3 = P + 0.5 * dr * k2_P
        
        if P3 < P_arr[0]:
            P3 = P_arr[0]
        if P3 > P_arr[-1]:
            P3 = P_arr[-1]
            
        rho3 = interp_rho(P3, rho_arr, P_arr)
        k3_m, k3_P = tov_derivatives_profile(r3, m3, P3, rho3)
        
        r4 = r + dr
        m4 = m + dr * k3_m
        P4 = P + dr * k3_P
        
        if P4 < P_arr[0]:
            P4 = P_arr[0]
        if P4 > P_arr[-1]:
            P4 = P_arr[-1]
            
        rho4 = interp_rho(P4, rho_arr, P_arr)
        k4_m, k4_P = tov_derivatives_profile(r4, m4, P4, rho4)
        
        # Update
        r += dr
        m += dr * (k1_m + 2*k2_m + 2*k3_m + k4_m) / 6.0
        P += dr * (k1_P + 2*k2_P + 2*k3_P + k4_P) / 6.0
        
        # Stop if pressure drops below surface threshold
        if P <= P_SURF or P < 0:
            break
        
        # Safety check on mass growth
        if m < 0 or not np.isfinite(m):
            break
        
        # Store values
        if P >= P_arr[0] and P <= P_arr[-1]:
            rho_arr_out[idx] = interp_rho(P, rho_arr, P_arr)
        else:
            rho_arr_out[idx] = 0.0
            
        r_arr_out[idx] = r
        m_arr_out[idx] = m
        P_arr_out[idx] = P
        idx += 1
    
    # Return only filled portion
    return (r_arr_out[:idx], m_arr_out[:idx], P_arr_out[:idx], 
            rho_arr_out[:idx], r, m)

def find_rho_c_for_mass(target_mass_solar, rho_arr, P_arr, M_rho_grid):
    """
    Find central density for target mass
    Uses the existing working solver
    """
    rho_grid, M_grid = M_rho_grid
    
    # Find maximum mass
    max_idx = np.argmax(M_grid)
    M_max = M_grid[max_idx]
    M_min = M_grid[0]
    
    print(f"  EOS mass range: [{M_min:.3f}, {M_max:.3f}] M☉")
    print(f"  Target mass: {target_mass_solar:.3f} M☉")
    
    if target_mass_solar > M_max:
        raise ValueError(f"Target mass {target_mass_solar:.3f} M☉ exceeds maximum {M_max:.3f} M☉")
    
    if target_mass_solar < M_min:
        raise ValueError(f"Target mass {target_mass_solar:.3f} M☉ is below minimum {M_min:.3f} M☉")
    
    # Use stable branch only
    stable_rho = rho_grid[:max_idx+1]
    stable_M = M_grid[:max_idx+1]
    
    # Find bracketing indices
    idx = np.searchsorted(stable_M, target_mass_solar)
    
    if idx == 0:
        rho_low = stable_rho[0]
        rho_high = stable_rho[min(1, len(stable_rho)-1)]
    elif idx >= len(stable_M):
        rho_low = stable_rho[-2]
        rho_high = stable_rho[-1]
    else:
        rho_low = stable_rho[idx-1]
        rho_high = stable_rho[idx]
    
    print(f"  Bracket: ρ ∈ [{rho_low:.3e}, {rho_high:.3e}] g/cm³")
    
    # Define objective function using existing solver
    def mass_difference(rho_c):
        r, m, P = ns_solve_numba(rho_c, rho_arr, P_arr)
        M_solar = m / 2e33
        diff = M_solar - target_mass_solar
        return diff
    
    # Check bracket validity
    f_low = mass_difference(rho_low)
    f_high = mass_difference(rho_high)
    
    print(f"  f(ρ_low) = {f_low:.6f} M☉")
    print(f"  f(ρ_high) = {f_high:.6f} M☉")
    
    if f_low * f_high > 0:
        # Try to expand bracket
        print(f"  Expanding bracket...")
        if abs(f_low) < abs(f_high):
            rho_low = rho_low * 0.8
            f_low = mass_difference(rho_low)
        else:
            rho_high = rho_high * 1.2
            f_high = mass_difference(rho_high)
        
        if f_low * f_high > 0:
            raise ValueError(
                f"Cannot bracket target mass. "
                f"M(ρ_low) = {f_low + target_mass_solar:.3f} M☉, "
                f"M(ρ_high) = {f_high + target_mass_solar:.3f} M☉"
            )
    
    # Use Brent's method
    print(f"  Running root finder...")
    rho_c = brentq(mass_difference, rho_low, rho_high, xtol=1e12, maxiter=100)
    print(f"  Found ρc = {rho_c:.3e} g/cm³")
    
    return rho_c


def compute_density_profile(eos_name, target_mass_solar):
    """
    Compute density profile for a specific neutron star mass
    """
    if eos_name not in eos_cache:
        raise ValueError(f"EOS {eos_name} not found")
    
    rho_eos = eos_cache[eos_name]['rho']
    P_eos = eos_cache[eos_name]['P']
    
    print(f"\nComputing profile for {eos_name}, M = {target_mass_solar:.3f} M☉")
    
    # Step 1: Build M(ρc) curve using existing solver
    print(f"Building M(ρc) curve...")
    rho_grid = np.concatenate([
        np.arange(2.5e14, 1e15, 0.5e13),
        np.arange(1e15, 4e15, 0.5e14)
    ])
    
    # Use your existing fast solver
    results = solve_all_densities(rho_grid, rho_eos, P_eos)
    M_grid = results[1, :] / 2e33  # Convert to solar masses
    
    print(f"  Computed {len(M_grid)} mass points")
    print(f"  Mass range: [{np.min(M_grid):.3f}, {np.max(M_grid):.3f}] M☉")
    
    # Step 2: Find ρc for target mass
    rho_c = find_rho_c_for_mass_v3(
        target_mass_solar,
        rho_eos,
        P_eos,
        (rho_grid, M_grid)
    )
    
    # Step 3: Integrate with profile storage
    print(f"Integrating with profile storage...")
    r_arr, m_arr, P_arr, rho_arr, R_final, M_final = ns_solve_profile(
        rho_c, rho_eos, P_eos
    )
    
    # Convert units
    r_km = r_arr / 1e5  # cm to km
    M_solar = m_arr / 2e33  # g to solar masses
    R_km = R_final / 1e5
    M_solar_final = M_final / 2e33
    
    print(f"Profile complete:")
    print(f"  R = {R_km:.2f} km")
    print(f"  M = {M_solar_final:.3f} M☉")
    print(f"  Points: {len(r_km)}\n")
    
    return {
        'eos_name': eos_name,
        'target_mass': target_mass_solar,
        'rho_c': float(rho_c),
        'R': float(R_km),
        'M': float(M_solar_final),
        'r': r_km.tolist(),
        'rho': rho_arr.tolist(),
        'P': P_arr.tolist(),
        'M_r': M_solar.tolist()
    }

# Add this route to your Flask app
@app.route('/profile', methods=['POST'])
def compute_profile():
    """Compute density profile for specific mass"""
    try:
        data = request.get_json()
        eos_name = data.get('eos')
        target_mass = float(data.get('mass', 1.4))
        
        print(f"\n{'='*60}")
        print(f"DENSITY PROFILE REQUEST")
        print(f"  EOS: {eos_name}")
        print(f"  Target mass: {target_mass} M☉")
        print(f"{'='*60}")
        
        if not eos_name or eos_name not in eos_cache:
            return jsonify({'error': 'Invalid EOS selected'}), 400
        
        # Compute profile
        profile = compute_density_profile(eos_name, target_mass)
        
        # Create density profile plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['rho'],
            mode='lines',
            name='ρ(r)',
            line=dict(color='#00D9FF', width=3)
        ))
        fig.update_layout(
            title=f"Density Profile: {eos_name} (M = {target_mass:.3f} M☉, ρc = {profile['rho_c']:.2e} g/cm³)",
            xaxis_title='Radius (km)',
            yaxis_title='Density ρ (g/cm³)',
            yaxis_type='log',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Mass profile plot
        fig_mass = go.Figure()
        fig_mass.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['M_r'],
            mode='lines',
            name='M(r)',
            line=dict(color='#FF6B6B', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 107, 0.2)'
        ))
        fig_mass.update_layout(
            title=f"Enclosed Mass Profile: {eos_name} (R = {profile['R']:.2f} km)",
            xaxis_title='Radius (km)',
            yaxis_title='Enclosed Mass M(r) (M☉)',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        mass_plot_json = json.dumps(fig_mass, cls=plotly.utils.PlotlyJSONEncoder)
        
        print(f"{'='*60}")
        print(f"SUCCESS - Profile computed")
        print(f"{'='*60}\n")
        
        return jsonify({
            'profile': profile,
            'density_plot': plot_json,
            'mass_plot': mass_plot_json
        })
    
    except ValueError as e:
        print(f"VALUE ERROR: {str(e)}\n")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        return jsonify({'error': str(e)}), 500
    
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
            print(f"    ✓ Max mass: {result['max_mass']:.3f} M☉")
            print(f"    ✓ Data points: {len(result['M'])}")
        
        if not results:
            return jsonify({'error': 'No valid EOS could be computed'}), 400
        
        # Debug inertia specifically
        if plot_type == 'inertia':
            print(f"DEBUG: Creating inertia plot with {len(results)} results")
            for i, result in enumerate(results):
                M = np.array(result['M'])
                R = np.array(result['R'])
                max_idx = np.argmax(M)
                print(f"  Result {i}: {result['name']}")
                print(f"    Total points: {len(M)}, Stable branch: {max_idx+1}")
                print(f"    M range: {M.min():.3f} to {M.max():.3f}")
                print(f"    R range: {R.min():.3f} to {R.max():.3f}")
        
        # Create plot
        plot_json = create_plot(results, plot_type)
        
        # Prepare statistics
        stats = [{
            'name': r['name'],
            'max_mass': f"{r['max_mass']:.3f}",
            'radius': f"{r['max_radius']:.2f}"
        } for r in results]
        
        print(f"  ✓ Computation complete - Plot type: {plot_type}\n")
        
        return jsonify({
            'plot': plot_json,
            'stats': stats,
            'results': results
        })
    
    except Exception as e:
        print(f"ERROR in compute: {str(e)}")
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

@app.route('/download', methods=['POST'])
def download():
    """Download results as CSV"""
    data = request.get_json()
    stats = data.get('stats', [])
    
    csv_content = "EOS,Max Mass (M☉),Radius (km)\n"
    for stat in stats:
        csv_content += f"{stat['name']},{stat['max_mass']},{stat['radius']}\n"
    
    return send_file(
        io.BytesIO(csv_content.encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='neutron_star_results.csv'
    )

# HTML Template
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neutron Star Solver</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { 
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            min-height: 100vh;
        }
        .dark-card {
            background: rgba(30, 41, 59, 0.8);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(71, 85, 105, 0.5);
        }
        .dark-card:hover {
            border-color: rgba(100, 116, 139, 0.7);
            transition: border-color 0.3s ease;
        }
        .checkbox-label:hover {
            background: rgba(51, 65, 85, 0.5);
        }
        .scrollbar-custom::-webkit-scrollbar {
            width: 6px;
        }
        .scrollbar-custom::-webkit-scrollbar-track {
            background: rgba(30, 41, 59, 0.5);
            border-radius: 3px;
        }
        .scrollbar-custom::-webkit-scrollbar-thumb {
            background: rgba(100, 116, 139, 0.6);
            border-radius: 3px;
        }
        .scrollbar-custom::-webkit-scrollbar-thumb:hover {
            background: rgba(148, 163, 184, 0.8);
        }
    </style>
</head>
<body class="text-gray-100 p-4 md:p-8">
    <div class="max-w-7xl mx-auto">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl md:text-5xl font-bold mb-3 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-400">
                Neutron Star Solver
            </h1>
            <p class="text-lg text-gray-400 max-w-2xl mx-auto">
                TOV Equation Solver for Various Equations of State - Explore the properties of neutron stars
            </p>
        </div>
        
        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            <!-- EOS Selection -->
            <div class="dark-card rounded-xl p-6 shadow-2xl">
                <h2 class="text-xl font-semibold mb-4 text-cyan-300">Select EOS</h2>
                <div id="eos-list" class="space-y-2 mb-4 max-h-96 overflow-y-auto scrollbar-custom pr-2">
                    {% for eos in eos_list %}
                    <label class="checkbox-label flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all duration-200">
                        <input type="checkbox" value="{{ eos.name }}" class="eos-checkbox w-5 h-5 rounded border-gray-600 bg-slate-700 text-cyan-400 focus:ring-cyan-400">
                        <span class="w-6 h-6 rounded border-2 border-gray-400" style="background-color: {{ eos.color }}"></span>
                        <span class="font-medium text-gray-200">{{ eos.name }}</span>
                    </label>
                    {% endfor %}
                </div>
                
                <!-- File Upload -->
                <div class="mt-6 pt-4 border-t border-slate-600">
                    <label class="flex flex-col items-center justify-center gap-3 p-4 bg-slate-700/50 hover:bg-slate-600/50 rounded-lg cursor-pointer transition-all duration-200 border-2 border-dashed border-slate-500 hover:border-cyan-400">
                        <svg class="w-8 h-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                        </svg>
                        <span class="font-medium text-cyan-300">Upload Custom EOS</span>
                        <input type="file" id="file-upload" accept=".csv" class="hidden">
                    </label>
                    <p class="text-xs text-gray-500 mt-2 text-center">CSV format with columns: rho, P</p>
                </div>
            </div>
            
            <!-- Plot Selection -->
            <div class="dark-card rounded-xl p-6 shadow-2xl">
                <h2 class="text-xl font-semibold mb-4 text-cyan-300">Plot Type</h2>
                <div class="space-y-3">
                    <label class="checkbox-label flex items-center gap-3 p-4 rounded-lg cursor-pointer transition-all duration-200">
                        <input type="radio" name="plot" value="mass-rho" class="w-5 h-5 text-cyan-400 focus:ring-cyan-400">
                        <span class="font-medium text-gray-200">Mass vs Central Density</span>
                    </label>
                    <label class="checkbox-label flex items-center gap-3 p-4 rounded-lg cursor-pointer transition-all duration-200">
                        <input type="radio" name="plot" value="radius-rho" class="w-5 h-5 text-cyan-400 focus:ring-cyan-400">
                        <span class="font-medium text-gray-200">Radius vs Central Density</span>
                    </label>
                    <label class="checkbox-label flex items-center gap-3 p-4 rounded-lg cursor-pointer transition-all duration-200">
                        <input type="radio" name="plot" value="mass-radius" class="w-5 h-5 text-cyan-400 focus:ring-cyan-400" checked>
                        <span class="font-medium text-gray-200">Mass vs Radius</span>
                    </label>
                    <label class="checkbox-label flex items-center gap-3 p-4 rounded-lg cursor-pointer transition-all duration-200">
                        <input type="radio" name="plot" value="inertia" class="w-5 h-5 text-cyan-400 focus:ring-cyan-400">
                        <span class="font-medium text-gray-200">Moment of Inertia</span>
                    </label>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="dark-card rounded-xl p-6 shadow-2xl">
                <h2 class="text-xl font-semibold mb-4 text-cyan-300">Statistics</h2>
                <div id="stats-container" class="space-y-3 max-h-96 overflow-y-auto scrollbar-custom pr-2">
                    <div class="text-center py-8">
                        <svg class="w-12 h-12 text-gray-600 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
                        </svg>
                        <p class="text-gray-500 text-sm">Run computation to see results</p>
                    </div>
                </div>
                <button id="download-btn" class="w-full mt-6 flex items-center justify-center gap-3 p-3 bg-emerald-600 hover:bg-emerald-500 rounded-lg font-medium transition-all duration-200 hidden">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    <span>Download CSV</span>
                </button>
            </div>
        </div>
        <!-- Density Profile Section -->
        <div class="dark-card rounded-xl p-6 shadow-2xl lg:col-span-3">
            <h2 class="text-xl font-semibold mb-4 text-cyan-300">Density Profile</h2>
            <p class="text-sm text-gray-400 mb-4">
                Compute the radial density distribution ρ(r) for a neutron star of specific mass
            </p>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Select EOS</label>
                    <select id="profile-eos" class="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-gray-200 focus:ring-2 focus:ring-cyan-400 focus:border-transparent">
                        <option value="">-- Select EOS --</option>
                        {% for eos in eos_list %}
                        <option value="{{ eos.name }}">{{ eos.name }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">
                        Target Mass (M☉)
                        <span class="text-xs text-gray-500">typically 1.0 - 2.5</span>
                    </label>
                    <input 
                        type="number" 
                        id="profile-mass" 
                        value="1.4" 
                        min="0.5" 
                        max="3.0" 
                        step="0.1"
                        class="w-full bg-slate-700 border border-slate-600 rounded-lg px-4 py-2 text-gray-200 focus:ring-2 focus:ring-cyan-400 focus:border-transparent"
                    >
                </div>
                
                <div class="flex items-end">
                    <button 
                        id="compute-profile-btn" 
                        class="w-full px-6 py-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 rounded-lg font-medium transition-all duration-200 transform hover:scale-105"
                    >
                        Compute Profile
                    </button>
                </div>
            </div>
            
            <div id="profile-loading" class="hidden text-center py-4">
                <div class="inline-flex items-center gap-3 bg-slate-800/80 rounded-full px-6 py-3">
                    <div class="animate-spin rounded-full h-5 w-5 border-b-2 border-emerald-400"></div>
                    <p class="text-emerald-300 text-sm font-medium">Computing density profile...</p>
                </div>
            </div>
            
            <div id="profile-results" class="hidden mt-6">
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div class="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                        <div class="text-xs text-gray-400 mb-1">Central Density</div>
                        <div id="profile-rho-c" class="text-lg font-bold text-cyan-300">--</div>
                    </div>
                    <div class="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                        <div class="text-xs text-gray-400 mb-1">Radius</div>
                        <div id="profile-radius" class="text-lg font-bold text-cyan-300">--</div>
                    </div>
                    <div class="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                        <div class="text-xs text-gray-400 mb-1">Final Mass</div>
                        <div id="profile-mass-final" class="text-lg font-bold text-cyan-300">--</div>
                    </div>
                    <div class="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                        <div class="text-xs text-gray-400 mb-1">Profile Points</div>
                        <div id="profile-points" class="text-lg font-bold text-cyan-300">--</div>
                    </div>
                </div>
                
                <div class="space-y-4">
                    <div id="density-profile-plot" class="bg-slate-800/50 rounded-lg p-4 border border-slate-600"></div>
                    <div id="mass-profile-plot" class="bg-slate-800/50 rounded-lg p-4 border border-slate-600"></div>
                </div>
            </div>
            
            <div id="profile-error" class="hidden mt-4 p-4 bg-red-900/30 border border-red-700 rounded-lg">
                <div class="flex items-start gap-3">
                    <svg class="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <div>
                        <div class="font-medium text-red-300 mb-1">Error Computing Profile</div>
                        <div id="profile-error-msg" class="text-sm text-red-200"></div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Compute Button -->
        <div class="mb-8 text-center">
            <button id="compute-btn" class="px-10 py-4 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 rounded-xl font-bold text-lg transition-all duration-300 transform hover:scale-105 shadow-lg">
                Solve TOV Equations
            </button>
            <div id="loading" class="hidden mt-6">
                <div class="inline-flex items-center gap-3 bg-slate-800/80 rounded-full px-6 py-3">
                    <div class="animate-spin rounded-full h-6 w-6 border-b-2 border-cyan-400"></div>
                    <p class="text-cyan-300 font-medium">Computing neutron star properties...</p>
                </div>
            </div>
        </div>
        
        <!-- Plot Area -->
        <div id="plot-container" class="dark-card rounded-xl p-4 md:p-6 border-2 border-slate-600 hidden mb-8">
            <div id="plot" class="w-full"></div>
        </div>
        
        <!-- Footer -->
        <div class="text-center text-gray-500 text-sm mt-8 pt-6 border-t border-slate-800">
            <p>Neutron Star Solver • TOV Equation Integration • Dark Mode</p>
        </div>
    </div>
    
    <script>
        let currentStats = [];
        
        console.log('Neutron Star Solver initialized');
        
        document.getElementById('compute-btn').addEventListener('click', function() {
            console.log('Compute button clicked');
            
            const selectedEOS = Array.from(document.querySelectorAll('.eos-checkbox:checked')).map(cb => cb.value);
            const plotType = document.querySelector('input[name="plot"]:checked').value;
            
            console.log('Selected EOS:', selectedEOS);
            console.log('Plot type:', plotType);
            
            if (selectedEOS.length === 0) {
                alert('Please select at least one EOS');
                return;
            }
            
            // Show loading state
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('compute-btn').disabled = true;
            document.getElementById('compute-btn').classList.add('opacity-50');
            
            fetch('/compute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ eos: selectedEOS, plot_type: plotType })
            })
            .then(response => {
                console.log('Response status:', response.status);
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Computation completed successfully');
                
                // Update plot
                Plotly.newPlot('plot', JSON.parse(data.plot).data, JSON.parse(data.plot).layout);
                document.getElementById('plot-container').classList.remove('hidden');
                
                // Update stats
                currentStats = data.stats;
                const statsHtml = data.stats.map(s => {
                    return `<div class="p-4 bg-slate-700/30 rounded-lg border border-slate-600">
                                <div class="font-bold text-cyan-300 text-sm mb-1">${s.name}</div>
                                <div class="text-xs text-gray-400">M<sub>max</sub>: ${s.max_mass} M<sub>☉</sub> • Radius: ${s.radius} km</div>
                            </div>`;
                }).join('');
                document.getElementById('stats-container').innerHTML = statsHtml;
                document.getElementById('download-btn').classList.remove('hidden');
                
                // Hide loading
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
                document.getElementById('compute-btn').classList.remove('opacity-50');
            })
            .catch(error => {
                console.error('Computation error:', error);
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
                document.getElementById('compute-btn').classList.remove('opacity-50');
            });
        });
        
        document.getElementById('file-upload').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const eosList = document.getElementById('eos-list');
                    const newLabel = document.createElement('label');
                    newLabel.className = 'checkbox-label flex items-center gap-3 p-3 rounded-lg cursor-pointer transition-all duration-200';
                    newLabel.innerHTML = `
                        <input type="checkbox" value="${data.name}" class="eos-checkbox w-5 h-5 rounded border-gray-600 bg-slate-700 text-cyan-400 focus:ring-cyan-400" checked>
                        <span class="w-6 h-6 rounded border-2 border-gray-400" style="background-color: ${data.color}"></span>
                        <span class="font-medium text-gray-200">${data.name} <span class="text-cyan-400 text-sm">(Custom)</span></span>
                    `;
                    eosList.appendChild(newLabel);
                    alert('Custom EOS uploaded successfully!');
                } else {
                    alert('Upload Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Upload error: ' + error.message);
            });
        });
        
        document.getElementById('download-btn').addEventListener('click', function() {
            fetch('/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stats: currentStats })
            })
            .then(response => response.blob())
            .then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'neutron_star_results.csv';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            });
        });
        document.getElementById('compute-profile-btn').addEventListener('click', function() {
            const eos = document.getElementById('profile-eos').value;
            const mass = parseFloat(document.getElementById('profile-mass').value);
            
            // Validation
            if (!eos) {
                alert('Please select an EOS');
                return;
            }
            
            if (isNaN(mass) || mass < 0.5 || mass > 3.0) {
                alert('Please enter a valid mass between 0.5 and 3.0 M☉');
                return;
            }
            
            // Hide previous results/errors
            document.getElementById('profile-results').classList.add('hidden');
            document.getElementById('profile-error').classList.add('hidden');
            
            // Show loading
            document.getElementById('profile-loading').classList.remove('hidden');
            document.getElementById('compute-profile-btn').disabled = true;
            
            console.log(`Computing density profile: ${eos}, M = ${mass} M☉`);
            
            fetch('/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ eos: eos, mass: mass })
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(data => {
                console.log('Profile computation successful');
                
                const profile = data.profile;
                
                // Update statistics
                document.getElementById('profile-rho-c').textContent = 
                    (profile.rho_c / 1e14).toFixed(2) + ' × 10¹⁴ g/cm³';
                document.getElementById('profile-radius').textContent = 
                    profile.R.toFixed(2) + ' km';
                document.getElementById('profile-mass-final').textContent = 
                    profile.M.toFixed(3) + ' M☉';
                document.getElementById('profile-points').textContent = 
                    profile.r.length;
                
                // Plot density profile
                Plotly.newPlot(
                    'density-profile-plot',
                    JSON.parse(data.density_plot).data,
                    JSON.parse(data.density_plot).layout,
                    {responsive: true}
                );
                
                // Plot mass profile
                Plotly.newPlot(
                    'mass-profile-plot',
                    JSON.parse(data.mass_plot).data,
                    JSON.parse(data.mass_plot).layout,
                    {responsive: true}
                );
                
                // Show results
                document.getElementById('profile-results').classList.remove('hidden');
                document.getElementById('profile-loading').classList.add('hidden');
                document.getElementById('compute-profile-btn').disabled = false;
                
                // Scroll to results
                document.getElementById('profile-results').scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });
            })
            .catch(error => {
                console.error('Profile computation error:', error);
                
                document.getElementById('profile-error-msg').textContent = error.message;
                document.getElementById('profile-error').classList.remove('hidden');
                document.getElementById('profile-loading').classList.add('hidden');
                document.getElementById('compute-profile-btn').disabled = false;
            });
        });
        
        console.log('UI initialization complete');
    </script>
</body>
</html>
'''

# Save template
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

# Initialize
print("Loading EOS data and compiling JIT functions...")
load_eos_data()

# Precompile Numba functions
dummy_rho = np.array([1e14, 2e14, 3e14], dtype=np.float64)
dummy_P = np.array([1e33, 2e33, 3e33], dtype=np.float64)
_ = ns_solve_numba(1.5e14, dummy_rho, dummy_P)
_ = solve_all_densities(np.array([1.5e14, 2e14]), dummy_rho, dummy_P)

print("✓ Ready!")
print("\n" + "="*50)
print("Starting Flask server...")
print("Access the app at: http://localhost:5000")
print("="*50 + "\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)