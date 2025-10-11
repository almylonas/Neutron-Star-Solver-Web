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
        'max_radius': float(R[np.argmax(M)])
    }

def compute_inertia(m, r):
    """Compute moment of inertia"""
    return (0.237 + 0.008) * m * r**2 * 1.989e43 * (1 + 4.2*(m/r) + 90*(m/r)**4) / 1e45

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
            max_idx = np.argmax(M)
            M_stable = M[:max_idx+1]
            R_stable = R[:max_idx+1]
            I = compute_inertia(M_stable, R_stable)
            
            fig.add_trace(go.Scatter(
                x=M_stable,
                y=I,
                mode='lines',
                name=result['name'],
                line=dict(color=result.get('color', '#000000'), width=2)
            ))
        fig.update_layout(
            title='Moment of Inertia',
            xaxis_title='Mass (M☉)',
            yaxis_title='I (10⁴⁵ g·cm²)',
            xaxis=dict(range=[1, 2.7]),
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
        body { background: linear-gradient(135 deg, rgb(30, 41, 59) 0%, rgb(88, 28, 135) 50%, rgb(30, 41, 59) 100%); }
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
            
            <!-- Plot Selection -->
            <div class="bg-slate-800/50 backdrop-blur rounded-lg p-6 border border-slate-700">
                <h2 class="text-xl font-semibold mb-4">Plot Type</h2>
                <div class="space-y-2">
                    <label class="flex items-center gap-3 p-3 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="mass-rho" class="w-4 h-4">
                        <span>Mass vs Central Density</span>
                    </label>
                    <label class="flex items-center gap-3 p-3 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="radius-rho" class="w-4 h-4">
                        <span>Radius vs Central Density</span>
                    </label>
                    <label class="flex items-center gap-3 p-3 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="mass-radius" class="w-4 h-4" checked>
                        <span>Mass vs Radius</span>
                    </label>
                    <label class="flex items-center gap-3 p-3 hover:bg-slate-700/50 rounded cursor-pointer">
                        <input type="radio" name="plot" value="inertia" class="w-4 h-4">
                        <span>Moment of Inertia</span>
                    </label>
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
                Solve TOV Equations
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
        
        document.getElementById('compute-btn').addEventListener('click', function() {
            console.log('Button clicked!');
            
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
_ = solve_all_densities(np.array([1.5e14, 2e14]), dummy_rho, dummy_P)

print("✓ Ready!")
print("\n" + "="*50)
print("Starting Flask server...")
print("Access the app at: http://localhost:5000")
print("="*50 + "\n")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)