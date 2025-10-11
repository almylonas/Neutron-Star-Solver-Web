<script>
        let currentStats = [];
        
        console.log('Neutron Star Solver initialized');
        
        // Show/hide profile options based on plot selection
        document.querySelectorAll('input[name="plot"]').forEach(radio => {
            radio.addEventListener('change', function() {
                const profileOptions = document.getElementById('profile-options');
                if (this.value === 'density-profile') {
                    profileOptions.classList.remove('hidden');
                } else {
                    profileOptions.classList.add('hidden');
                }
            });
        });
        
        document.getElementById('compute-btn').addEventListener('click', function() {
            console.log('Compute button clicked');
            
            const plotType = document.querySelector('input[name="plot"]:checked').value;
            
            if (plotType === 'density-profile') {
                computeDensityProfile();
            } else {
                computeMRCurves();
            }
        });
        
        function computeMRCurves() {"""
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

# EOS Configuration
EOS_CONFIG = [
    ("BSK20", "bsk20.csv", 6, "#000000"),
    ("DD2", "dd2.csv", 4, "#228B22"),
    ("APR3", "apr3.csv", 1, "#800080"),
    ("ENG", "eng.csv", 1, "#000080"),
    ("TMA", "tma.csv", 3, "#B22222"),
    ("WFF2", "wff2.csv", 1, "#00FFFF"),
    ("H4", "h4.csv", 1, "#FF69B4"),
    ("MPA1", "mpa1.csv", 4, "#CD853F"),
    ("SFHOY", "sfhoy.csv", 4, "#708090"),
    ("WFF1", "wff1.csv", 1, "#8FBC8F"),
    ("SFHO", "sfho.csv", 4, "#2F4F4F"),
    ("TM1", "tm1.csv", 4, "#FFE4B5"),
    ("APRLD", "aprld.csv", 4, "#D8BFD8"),
    ("BHBLP", "bhblp.csv", 4, "#4682B4"),
    ("DD2F", "dd2f.csv", 1, "#EE82EE"),
    ("SLY", "sly.csv", 1, "#CD5C5C"),
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
    dm_dr = FOUR_PI * r * r * rho * c2  # Include c^2 for energy density
    
    # TOV equation for pressure (using energy density epsilon = rho * c^2)
    epsilon = rho * c2
    numerator = -G * (epsilon + P) * (m + FOUR_PI * r * r * r * P / c2)
    denominator = r * (r - TWO_G_C2 * m)
    
    dP_dr = numerator / denominator
    
    return dr_dr, dm_dr, dP_dr

@jit(nopython=True, fastmath=True)
def ns_solve_numba_profile(rho_0, rho_arr, P_arr, dr=500.0, max_iter=80000, tol=1e-10):
    """Solve TOV and return full profile arrays"""
    P_0 = interp_P(rho_0, rho_arr, P_arr)
    
    r = 500.0
    m = (4.0/3.0) * np.pi * r * r * r * rho_0 * c2  # Initial mass with c^2
    P = P_0
    
    # Arrays to store profile
    r_arr = np.zeros(max_iter)
    m_arr = np.zeros(max_iter)
    P_arr_out = np.zeros(max_iter)
    
    r_arr[0] = r
    m_arr[0] = m
    P_arr_out[0] = P
    
    idx = 1
    
    for i in range(1, max_iter):
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
        
        r_arr[idx] = r
        m_arr[idx] = m
        P_arr_out[idx] = P
        idx += 1
        
        if P / P_0 < tol or P < 0:
            break
    
    return r_arr[:idx], m_arr[:idx], P_arr_out[:idx]

@jit(nopython=True, fastmath=True)
def ns_solve_numba(rho_0, rho_arr, P_arr, dr=500.0, max_iter=80000, tol=1e-10):
    """Fast solver returning only final values"""
    r_arr, m_arr, P_arr_out = ns_solve_numba_profile(rho_0, rho_arr, P_arr, dr, max_iter, tol)
    return r_arr[-1], m_arr[-1], P_arr_out[-1]

def find_rho_for_mass(target_mass_solar, rho_arr_eos, P_arr_eos, tol=1e-4, max_iter=50):
    """Find central density that produces target mass using bisection"""
    # Convert target mass to CGS (grams)
    target_mass = target_mass_solar * 2e33
    
    # First, find the range of rho_c that brackets the target mass
    rho_min = 1e14
    rho_max = 5e15
    
    # Refine brackets
    for _ in range(10):
        rho_test = (rho_min + rho_max) / 2
        r, m, P = ns_solve_numba(rho_test, rho_arr_eos, P_arr_eos)
        
        if m / 2e33 < target_mass_solar:
            rho_min = rho_test
        else:
            rho_max = rho_test
    
    # Bisection to find exact rho_c
    for iteration in range(max_iter):
        rho_c = (rho_min + rho_max) / 2
        r, m, P = ns_solve_numba(rho_c, rho_arr_eos, P_arr_eos)
        mass_solar = m / 2e33
        
        if abs(mass_solar - target_mass_solar) < tol:
            return rho_c
        
        if mass_solar < target_mass_solar:
            rho_min = rho_c
        else:
            rho_max = rho_c
    
    return rho_c
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

def find_rho_for_mass(target_mass_solar, rho_arr_eos, P_arr_eos, tol=1e-4, max_iter=50):
    """Find central density that produces target mass using bisection method"""
    # Convert target mass to CGS (grams)
    target_mass = target_mass_solar * 2e33
    
    # Initial brackets for rho_c
    rho_min = 1e14
    rho_max = 5e15
    
    # Refine brackets to ensure target mass is within range
    for _ in range(10):
        rho_test = (rho_min + rho_max) / 2
        r, m, P = ns_solve_numba(rho_test, rho_arr_eos, P_arr_eos)
        
        if m / 2e33 < target_mass_solar:
            rho_min = rho_test
        else:
            rho_max = rho_test
    
    # Bisection to find exact rho_c
    for iteration in range(max_iter):
        rho_c = (rho_min + rho_max) / 2
        r, m, P = ns_solve_numba(rho_c, rho_arr_eos, P_arr_eos)
        mass_solar = m / 2e33
        
        if abs(mass_solar - target_mass_solar) < tol:
            return rho_c
        
        if mass_solar < target_mass_solar:
            rho_min = rho_c
        else:
            rho_max = rho_c
    
    return rho_c

def compute_density_profile(target_mass_solar, eos_name, rho_arr, P_arr):
    """Compute density profile for a given target mass"""
    print(f"  Finding rho_c for M = {target_mass_solar} M_sun...")
    
    # Find central density
    rho_c = find_rho_for_mass(target_mass_solar, rho_arr, P_arr)
    
    print(f"    Found rho_c = {rho_c:.3e} g/cm^3")
    
    # Get full profile
    r_arr, m_arr, P_arr_out = ns_solve_numba_profile(rho_c, rho_arr, P_arr)
    
    # Convert pressure to density at each radius
    rho_arr_out = np.array([interp_rho(P, rho_arr, P_arr) for P in P_arr_out])
    
    # Convert to convenient units
    r_km = r_arr / 1e5  # cm to km
    rho_cgs = rho_arr_out  # g/cm^3
    m_solar = m_arr / 2e33  # g to solar masses
    
    return {
        'name': eos_name,
        'target_mass': target_mass_solar,
        'rho_c': float(rho_c),
        'radius': float(r_km[-1]),
        'r': r_km.tolist(),
        'rho': rho_cgs.tolist(),
        'm': m_solar.tolist(),
        'P': P_arr_out.tolist()
    }

def compute_density_profile(target_mass_solar, eos_name, rho_arr, P_arr):
    """Compute density profile for a given target mass"""
    print(f"  Finding rho_c for M = {target_mass_solar} M_sun...")
    
    # Find central density
    rho_c = find_rho_for_mass(target_mass_solar, rho_arr, P_arr)
    
    print(f"    Found rho_c = {rho_c:.3e} g/cm^3")
    
    # Get full profile
    r_arr, m_arr, P_arr_out = ns_solve_numba_profile(rho_c, rho_arr, P_arr)
    
    # Convert pressure to density
    rho_arr_out = np.array([interp_rho(P, rho_arr, P_arr) for P in P_arr_out])
    
    # Convert to convenient units
    r_km = r_arr / 1e5  # cm to km
    rho_cgs = rho_arr_out  # g/cm^3
    m_solar = m_arr / 2e33  # g to solar masses
    
    return {
        'name': eos_name,
        'target_mass': target_mass_solar,
        'rho_c': rho_c,
        'radius': float(r_km[-1]),
        'r': r_km.tolist(),
        'rho': rho_cgs.tolist(),
        'm': m_solar.tolist(),
        'P': P_arr_out.tolist()
    }

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
    """Compute moment of inertia with safety checks"""
    # Avoid division by zero and invalid values
    mask = (r > 0) & (m > 0) & np.isfinite(m) & np.isfinite(r)
    result = np.zeros_like(m)
    
    # Only compute for valid values
    valid_m = m[mask]
    valid_r = r[mask]
    
    # Compute m/r ratio
    m_over_r = valid_m / valid_r
    
    # Avoid overflow in the (m/r)^4 term
    m_over_r_safe = np.clip(m_over_r, 0, 10)  # Clip to reasonable range
    
    # I = (0.237 + 0.008) * M * R^2 * k * (1 + 4.2*(M/R) + 90*(M/R)^4)
    # k = 1.989e43 (conversion factor)
    # Result in units of 10^45 g·cm^2
    I_valid = (0.237 + 0.008) * valid_m * valid_r**2 * 1.989e43 * \
              (1 + 4.2 * m_over_r_safe + 90 * m_over_r_safe**4) / 1e45
    
    result[mask] = I_valid
    
    return result

def create_plot(results, plot_type):
    """Create Plotly figure based on plot type"""
    fig = go.Figure()
    
    if plot_type == 'mass-rho':
        for result in results:
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
        for result in results:
            M = np.array(result['M'])
            R = np.array(result['R'])
            
            # Find maximum mass index (stable branch)
            max_idx = np.argmax(M)
            
            # Only use stable branch (up to maximum mass)
            M_stable = M[:max_idx+1]
            R_stable = R[:max_idx+1]
            
            # Compute inertia
            I = compute_inertia(M_stable, R_stable)
            
            # Remove any NaN or Inf values
            valid_mask = np.isfinite(I) & (I > 0)
            M_plot = M_stable[valid_mask]
            I_plot = I[valid_mask]
            
            if len(M_plot) > 0:  # Only add trace if we have valid data
                fig.add_trace(go.Scatter(
                    x=M_plot,
                    y=I_plot,
                    mode='lines',
                    name=result['name'],
                    line=dict(color=result.get('color', '#000000'), width=2)
                ))
        
        fig.update_layout(
            title='Moment of Inertia',
            xaxis_title='Mass (M☉)',
            yaxis_title='I (10⁴⁵ g·cm²)',
            xaxis=dict(range=[0.5, 2.8]),
            yaxis=dict(range=[0, None]),  # Auto-range for y
            template='plotly_dark'
        )
    
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
        
        if not results:
            return jsonify({'error': 'No valid EOS could be computed'}), 400
        
        # Create plot
        plot_json = create_plot(results, plot_type)
        
        # Prepare statistics
        stats = [{
            'name': r['name'],
            'max_mass': f"{r['max_mass']:.3f}",
            'radius': f"{r['max_radius']:.2f}"
        } for r in results]
        
        print(f"  ✓ Computation complete\n")
        
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

@app.route('/compute_profile', methods=['POST'])
def compute_profile():
    """Compute density profile for a specific mass"""
    try:
        data = request.get_json()
        target_mass = float(data.get('mass', 1.4))
        eos_name = data.get('eos', 'DD2')
        
        print(f"\nProfile request:")
        print(f"  Target mass: {target_mass} M_sun")
        print(f"  EOS: {eos_name}")
        
        if eos_name not in eos_cache:
            return jsonify({'error': f'EOS {eos_name} not found'}), 400
        
        # Compute profile
        profile = compute_density_profile(
            target_mass,
            eos_name,
            eos_cache[eos_name]['rho'],
            eos_cache[eos_name]['P']
        )
        
        # Create plots
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['rho'],
            mode='lines',
            name=f'{eos_name} (M={target_mass} M☉)',
            line=dict(color=eos_cache[eos_name]['color'], width=2)
        ))
        fig_rho.update_layout(
            title=f'Density Profile for M = {target_mass} M☉',
            xaxis_title='Radius (km)',
            yaxis_title='Density (g/cm³)',
            yaxis_type='log',
            template='plotly_dark'
        )
        
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['m'],
            mode='lines',
            name=f'{eos_name} (M={target_mass} M☉)',
            line=dict(color=eos_cache[eos_name]['color'], width=2)
        ))
        fig_m.update_layout(
            title=f'Enclosed Mass Profile for M = {target_mass} M☉',
            xaxis_title='Radius (km)',
            yaxis_title='Enclosed Mass (M☉)',
            template='plotly_dark'
        )
        
        print(f"  ✓ Profile computed: R = {profile['radius']:.2f} km\n")
        
        return jsonify({
            'profile': profile,
            'plot_rho': json.dumps(fig_rho, cls=plotly.utils.PlotlyJSONEncoder),
            'plot_m': json.dumps(fig_m, cls=plotly.utils.PlotlyJSONEncoder)
        })
    
    except Exception as e:
        print(f"ERROR in compute_profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
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

@app.route('/compute_profile', methods=['POST'])
def compute_profile():
    """Compute density profile for a specific mass"""
    try:
        data = request.get_json()
        target_mass = float(data.get('mass', 1.4))
        eos_name = data.get('eos', 'DD2')
        
        print(f"\nProfile request:")
        print(f"  Target mass: {target_mass} M_sun")
        print(f"  EOS: {eos_name}")
        
        if eos_name not in eos_cache:
            return jsonify({'error': f'EOS {eos_name} not found'}), 400
        
        # Compute profile
        profile = compute_density_profile(
            target_mass,
            eos_name,
            eos_cache[eos_name]['rho'],
            eos_cache[eos_name]['P']
        )
        
        # Create density profile plot
        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['rho'],
            mode='lines',
            name=f'{eos_name}',
            line=dict(color=eos_cache[eos_name]['color'], width=3)
        ))
        fig_rho.update_layout(
            title=f'Density Profile: {eos_name} (M = {target_mass} M☉)',
            xaxis_title='Radius (km)',
            yaxis_title='Density (g/cm³)',
            yaxis_type='log',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        # Create mass profile plot
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=profile['r'],
            y=profile['m'],
            mode='lines',
            name=f'{eos_name}',
            line=dict(color=eos_cache[eos_name]['color'], width=3),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(eos_cache[eos_name]["color"][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}'
        ))
        fig_m.update_layout(
            title=f'Enclosed Mass Profile: {eos_name} (M = {target_mass} M☉)',
            xaxis_title='Radius (km)',
            yaxis_title='Enclosed Mass (M☉)',
            template='plotly_dark',
            hovermode='x unified'
        )
        
        print(f"  ✓ Profile computed: R = {profile['radius']:.2f} km, ρ_c = {profile['rho_c']:.2e} g/cm³\n")
        
        return jsonify({
            'profile': profile,
            'plot_rho': json.dumps(fig_rho, cls=plotly.utils.PlotlyJSONEncoder),
            'plot_m': json.dumps(fig_m, cls=plotly.utils.PlotlyJSONEncoder)
        })
    
    except Exception as e:
        print(f"ERROR in compute_profile: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
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

@app.route('/download', methods=['POST'])
def download():
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neutron Star Solver</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background: linear-gradient(135deg, rgb(30, 41, 59) 0%, rgb(88, 28, 135) 50%, rgb(30, 41, 59) 100%); }
    </style>
</head>
<body class="min-h-screen text-white p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-4xl font-bold mb-2 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Neutron Star Solver
        </h1>
        <p class="text-center text-gray-400 mb-8">TOV Equation Solver for Various Equations of State</p>
        
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <!-- EOS Selection -->
            <div class="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
                <h2 class="text-xl font-semibold mb-4">Select EOS</h2>
                <div id="eos-list" class="space-y-2 mb-4 max-h-96 overflow-y-auto">
                    {% for eos in eos_list %}
                    <label class="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="checkbox" value="{{ eos.name }}" class="eos-checkbox w-4 h-4">
                        <span class="w-6 h-6 rounded" style="background-color: {{ eos.color }}"></span>
                        <span>{{ eos.name }}</span>
                    </label>
                    {% endfor %}
                </div>
                
                <label class="flex items-center justify-center gap-2 p-3 bg-blue-600 hover:bg-blue-700 rounded cursor-pointer transition">
                    <span>Upload Custom EOS</span>
                    <input type="file" id="file-upload" accept=".csv" class="hidden">
                </label>
                <p class="text-xs text-gray-400 mt-2 text-center">CSV format: rho,P</p>
            </div>
            
            <!-- Plot Selection / Profile Input -->
            <div class="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
                <h2 class="text-xl font-semibold mb-4">Mode Selection</h2>
                
                <label class="flex items-center gap-3 p-3 bg-slate-700/50 rounded cursor-pointer mb-3">
                    <input type="radio" name="mode" value="mr-curves" class="w-4 h-4" checked>
                    <span>M-R Curves (Multiple EOS)</span>
                </label>
                
                <div id="mr-mode" class="space-y-2">
                    <p class="text-sm text-gray-400 mb-2">Plot Type:</p>
                    <label class="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="mass-rho" class="w-4 h-4">
                        <span class="text-sm">Mass vs Central Density</span>
                    </label>
                    <label class="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="radius-rho" class="w-4 h-4">
                        <span class="text-sm">Radius vs Central Density</span>
                    </label>
                    <label class="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="mass-radius" class="w-4 h-4" checked>
                        <span class="text-sm">Mass vs Radius</span>
                    </label>
                    <label class="flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="inertia" class="w-4 h-4">
                        <span class="text-sm">Moment of Inertia</span>
                    </label>
                </div>
                
                <label class="flex items-center gap-3 p-3 bg-slate-700/50 rounded cursor-pointer mt-3">
                    <input type="radio" name="mode" value="profile" class="w-4 h-4">
                    <span>Density Profile (Single Star)</span>
                </label>
                
                <div id="profile-mode" class="hidden mt-3 space-y-3">
                    <div>
                        <label class="block text-sm font-medium mb-1">Target Mass (M☉)</label>
                        <input type="number" id="target-mass" value="1.4" step="0.1" min="0.5" max="3.0" 
                               class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm">
                    </div>
                    <div>
                        <label class="block text-sm font-medium mb-1">EOS</label>
                        <select id="profile-eos" class="w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded text-white text-sm">
                            {% for eos in eos_list %}
                            <option value="{{ eos.name }}">{{ eos.name }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>
            </div>
            
            <!-- Statistics -->
            <div class="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
                <h2 class="text-xl font-semibold mb-4">Statistics</h2>
                <div id="stats-container" class="space-y-2 max-h-96 overflow-y-auto">
                    <p class="text-gray-400 text-sm">Run computation to see results</p>
                </div>
                <button id="download-btn" class="w-full mt-4 flex items-center justify-center gap-2 p-2 bg-green-600 hover:bg-green-700 rounded transition hidden">
                    <span>Download CSV</span>
                </button>
            </div>
        </div>
        
        <!-- Compute Button -->
        <div class="mb-6 text-center">
            <button id="compute-btn" class="px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 rounded-lg font-semibold text-lg transition">
                <span id="compute-btn-text">Solve TOV Equations</span>
            </button>
            <div id="loading" class="hidden mt-4">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                <p class="mt-2">Computing...</p>
            </div>
        </div>
        
        <!-- Plot Area -->
        <div id="plot-container" class="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700 hidden">
            <div id="plot"></div>
        </div>
    </div>
    
    <script>
        let currentStats = [];
        
        console.log('Script started');
        
        // Mode switching
        document.querySelectorAll('input[name="mode"]').forEach(function(radio) {
            radio.addEventListener('change', function() {
                const mrMode = document.getElementById('mr-mode');
                const profileMode = document.getElementById('profile-mode');
                const btnText = document.getElementById('compute-btn-text');
                
                if (this.value === 'mr-curves') {
                    mrMode.classList.remove('hidden');
                    profileMode.classList.add('hidden');
                    btnText.textContent = 'Solve TOV Equations';
                } else {
                    mrMode.classList.add('hidden');
                    profileMode.classList.remove('hidden');
                    btnText.textContent = 'Compute Density Profile';
                }
            });
        });
        
        document.getElementById('compute-btn').addEventListener('click', function() {
            console.log('Button clicked!');
            
            const mode = document.querySelector('input[name="mode"]:checked').value;
            
            if (mode === 'mr-curves') {
                computeMRCurves();
            } else {
                computeProfile();
            }
        });
        
        function computeMRCurves() {
            
            const selectedEOS = Array.from(document.querySelectorAll('.eos-checkbox:checked')).map(function(cb) { return cb.value; });
            const plotType = document.querySelector('input[name="plot"]:checked').value;
            
            console.log('Selected EOS:', selectedEOS);
            console.log('Plot type:', plotType);
            
            if (selectedEOS.length === 0) {
                alert('Please select at least one EOS');
                return;
            }
            
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('compute-btn').disabled = true;
            
            fetch('/compute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ eos: selectedEOS, plot_type: plotType })
            })
            .then(function(response) {
                console.log('Response received:', response.status);
                if (!response.ok) {
                    return response.json().then(function(errorData) {
                        throw new Error(errorData.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(function(data) {
                console.log('Data received:', data);
                
                // Update plot
                Plotly.newPlot('plot', JSON.parse(data.plot).data, JSON.parse(data.plot).layout);
                document.getElementById('plot-container').classList.remove('hidden');
                
                // Update stats
                currentStats = data.stats;
                const statsHtml = data.stats.map(function(s) {
                    return '<div class="p-2 bg-slate-700/30 rounded"><div class="font-semibold text-sm">' + s.name + '</div><div class="text-xs text-gray-400">M<sub>max</sub>: ' + s.max_mass + ' M<sub>sun</sub> @ R: ' + s.radius + ' km</div></div>';
                }).join('');
                document.getElementById('stats-container').innerHTML = statsHtml;
                document.getElementById('download-btn').classList.remove('hidden');
                
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
            })
            .catch(function(error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
            });
        }
        
        function computeProfile() {
            const targetMass = parseFloat(document.getElementById('target-mass').value);
            const eosName = document.getElementById('profile-eos').value;
            
            console.log('Computing profile for mass:', targetMass, 'EOS:', eosName);
            
            if (isNaN(targetMass) || targetMass <= 0) {
                alert('Please enter a valid target mass');
                return;
            }
            
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('compute-btn').disabled = true;
            document.getElementById('stats-container').innerHTML = '<p class="text-gray-400 text-sm">Computing...</p>';
            
            fetch('/compute_profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mass: targetMass, eos: eosName })
            })
            .then(function(response) {
                console.log('Profile response received:', response.status);
                if (!response.ok) {
                    return response.json().then(function(errorData) {
                        throw new Error(errorData.error || 'Server error');
                    });
                }
                return response.json();
            })
            .then(function(data) {
                console.log('Profile data received');
                
                // Update stats
                const profile = data.profile;
                const statsHtml = '<div class="p-2 bg-slate-700/30 rounded">' +
                    '<div class="font-semibold text-sm">' + profile.name + ' - M = ' + profile.target_mass + ' M<sub>sun</sub></div>' +
                    '<div class="text-xs text-gray-400">Central Density: ' + profile.rho_c.toExponential(2) + ' g/cm<sup>3</sup></div>' +
                    '<div class="text-xs text-gray-400">Radius: ' + profile.radius.toFixed(2) + ' km</div>' +
                    '</div>';
                document.getElementById('stats-container').innerHTML = statsHtml;
                
                // Create combined plot with two subplots
                var plotDiv = document.getElementById('plot');
                plotDiv.innerHTML = '';
                
                // Parse plots
                var plotRho = JSON.parse(data.plot_rho);
                var plotM = JSON.parse(data.plot_m);
                
                // Create subplot layout
                Plotly.newPlot('plot', 
                    [...plotRho.data, ...plotM.data],
                    {
                        grid: {rows: 1, columns: 2, pattern: 'independent'},
                        template: 'plotly_dark',
                        title: 'Density Profile for M = ' + profile.target_mass + ' M☉ (' + profile.name + ')',
                        xaxis: {title: 'Radius (km)', domain: [0, 0.45]},
                        yaxis: {title: 'Density (g/cm³)', type: 'log'},
                        xaxis2: {title: 'Radius (km)', domain: [0.55, 1]},
                        yaxis2: {title: 'Enclosed Mass (M☉)', anchor: 'x2'}
                    }
                );
                
                // Update trace xaxis/yaxis
                Plotly.restyle('plot', {'xaxis': 'x1', 'yaxis': 'y1'}, [0]);
                Plotly.restyle('plot', {'xaxis': 'x2', 'yaxis': 'y2'}, [1]);
                
                document.getElementById('plot-container').classList.remove('hidden');
                
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
            })
            .catch(function(error) {
                console.error('Profile error:', error);
                alert('Error: ' + error.message);
                document.getElementById('loading').classList.add('hidden');
                document.getElementById('compute-btn').disabled = false;
            });
        }
        
        document.getElementById('file-upload').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(function(response) { return response.json(); })
            .then(function(data) {
                if (data.success) {
                    const eosList = document.getElementById('eos-list');
                    const newLabel = document.createElement('label');
                    newLabel.className = 'flex items-center gap-3 p-2 hover:bg-slate-700/50 rounded cursor-pointer';
                    newLabel.innerHTML = '<input type="checkbox" value="' + data.name + '" class="eos-checkbox w-4 h-4" checked><span class="w-6 h-6 rounded" style="background-color: ' + data.color + '"></span><span>' + data.name + ' (Custom)</span>';
                    eosList.appendChild(newLabel);
                    alert('EOS uploaded successfully!');
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(function(error) {
                alert('Upload error: ' + error.message);
            });
        });
        
        document.getElementById('download-btn').addEventListener('click', function() {
            fetch('/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stats: currentStats })
            })
            .then(function(response) { return response.blob(); })
            .then(function(blob) {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'neutron_star_results.csv';
                a.click();
            });
        });
        
        console.log('Script finished loading');
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
_ = ns_solve_numba_profile(1.5e14, dummy_rho, dummy_P)
_ = solve_all_densities(np.array([1.5e14, 2e14]), dummy_rho, dummy_P)

print("✓ Ready!")
print("\n" + "="*50)
print("Starting Flask server...")
print("Access the app at: http://localhost:5000")
print("="*50 + "\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)