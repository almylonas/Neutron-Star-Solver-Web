"""
Neutron Star Solver Flask Web Application
Run with: python main.py          (development server)
      or: gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 main:app
Access at: http://localhost:5000
"""

import csv
import io
import os
import tempfile
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Serverless hosts (Vercel, AWS Lambda, ...) mount the code read-only and only
# allow writes to /tmp. Numba's on-disk compile cache would then fail at import
# ("no locator available"), so point it at the temp dir. Must happen before
# numba is imported.
if 'NUMBA_CACHE_DIR' not in os.environ and not os.access(BASE_DIR, os.W_OK):
    os.environ['NUMBA_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), 'numba_cache')

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from numba import njit, prange
from werkzeug.utils import secure_filename
EOS_DIR = os.path.join(BASE_DIR, 'eos')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ---------------------------------------------------------------------------
# Constants (CGS)
# ---------------------------------------------------------------------------
G = 6.67e-8
c = 3e10
c2 = c * c
TWO_G_C2 = 2 * G / c2
FOUR_PI = 4.0 * np.pi
M_SUN = 2e33            # g (value used throughout the original app)
RHO_SAT = 2.7e14        # nuclear saturation density, g/cm³ (reference line on ρ(r) plots)

# Solver settings
R_START = 100.0         # cm, inner boundary (m and P from the central Taylor expansion)
P_STOP_FRAC = 1e-10     # surface defined as P = 1e-10 · P_c (same criterion as before)
RTOL = 1e-8             # relative error tolerance of the adaptive integrator
H_MAX = 2e4             # cm, largest allowed step (200 m)
MAX_STEPS = 200000
PROFILE_MAX_POINTS = 20000

# Grid of central densities used for the M–R / M–ρc / R–ρc curves
RHO_C_GRID = np.concatenate([
    np.arange(2.5e14, 1e15, 0.5e13),
    np.arange(1e15, 4e15, 0.5e14)
])

# EOS Configuration: (name, file, sampling, colour)
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

eos_cache = {}          # name -> {'rho', 'P', 'color'}
_curve_cache = {}       # name -> computed M(ρc), R(ρc) curve
_profile_cache = {}     # (name, mode, value) -> (profile, warnings)
PROFILE_CACHE_SIZE = 256
_cache_lock = threading.Lock()
# Numba's default (workqueue) threading layer must not be entered from two
# threads at once, and the Flask/gunicorn servers are threaded -> serialise.
_solver_lock = threading.Lock()


def _clean_eos(rho, P):
    """Drop invalid rows and sort by density so the binary searches are valid."""
    rho = np.asarray(rho, dtype=np.float64)
    P = np.asarray(P, dtype=np.float64)
    ok = np.isfinite(rho) & np.isfinite(P) & (rho > 0) & (P > 0)
    rho, P = rho[ok], P[ok]
    order = np.argsort(rho, kind='stable')
    rho, P = rho[order], P[order]
    if len(rho) < 2:
        raise ValueError("EOS needs at least two valid (rho, P) rows")
    if np.any(np.diff(P) < 0):
        raise ValueError("Pressure must not decrease with density")
    return np.ascontiguousarray(rho), np.ascontiguousarray(P)


def _read_rho_P(text):
    """
    Read the 'rho' and 'P' columns of a CSV (other columns are ignored).
    Blank lines are skipped; unparsable cells become NaN (dropped later).
    Kept dependency-free (no pandas) to keep the serverless bundle small.
    """
    rows = [row for row in csv.reader(io.StringIO(text)) if any(c.strip() for c in row)]
    if not rows:
        raise ValueError("Empty file")
    header = [h.strip() for h in rows[0]]
    if 'rho' not in header or 'P' not in header:
        raise ValueError('CSV must have "rho" and "P" columns')
    i_rho, i_P = header.index('rho'), header.index('P')

    def num(row, i):
        try:
            return float(row[i])
        except (IndexError, ValueError):
            return np.nan

    data = rows[1:]
    return (np.array([num(r, i_rho) for r in data], dtype=np.float64),
            np.array([num(r, i_P) for r in data], dtype=np.float64))


def load_eos_data():
    """Load all predefined EOS data"""
    loaded_count = 0

    if not os.path.isdir(EOS_DIR):
        print(f"ERROR: EOS directory not found: {EOS_DIR}")
        return

    print(f"\nLoading EOS files from: {EOS_DIR}")
    for name, filename, sampling, color in EOS_CONFIG:
        filepath = os.path.join(EOS_DIR, filename)
        try:
            if not os.path.exists(filepath):
                print(f"  ✗ {filename} - FILE NOT FOUND")
                continue
            with open(filepath, encoding='utf-8-sig') as fh:
                rho, P = _read_rho_P(fh.read())
            if sampling > 1:
                rho, P = rho[1::sampling], P[1::sampling]
            rho, P = _clean_eos(rho, P)
            eos_cache[name] = {'rho': rho, 'P': P, 'color': color}
            loaded_count += 1
            print(f"  ✓ {filename} - Loaded successfully ({len(rho)} points)")
        except Exception as e:
            print(f"  ✗ {filename} - Error: {e}")

    print(f"\nTotal EOS loaded: {loaded_count}/{len(EOS_CONFIG)}")
    if loaded_count == 0:
        print("\nWARNING: No EOS files were loaded!")


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------
@njit(fastmath=True, cache=True)
def interp_lin(x, xs, ys):
    """Linear interpolation y(x) on a sorted table (clamped at the ends)."""
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


@njit(fastmath=True, cache=True, inline='always')
def tov_rhs(r, m, P, rho_t, P_t):
    """TOV right-hand side: returns (dm/dr, dP/dr)."""
    if P < 0.0:
        P = 0.0
    rho = interp_lin(P, P_t, rho_t)
    dm = FOUR_PI * r * r * rho
    dP = -G * (rho + P / c2) * (m + FOUR_PI * r * r * r * P / c2) / (
        r * r * (1.0 - TWO_G_C2 * m / r))
    return dm, dP


# Dormand–Prince 5(4) tableau
_A21 = 1.0 / 5.0
_A31, _A32 = 3.0 / 40.0, 9.0 / 40.0
_A41, _A42, _A43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
_A51, _A52, _A53, _A54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
_A61, _A62, _A63, _A64, _A65 = (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0,
                                49.0 / 176.0, -5103.0 / 18656.0)
_B1, _B3, _B4, _B5, _B6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0
_E1, _E3, _E4, _E5, _E6, _E7 = (71.0 / 57600.0, -71.0 / 16695.0, 71.0 / 1920.0,
                                -17253.0 / 339200.0, 22.0 / 525.0, -1.0 / 40.0)


@njit(fastmath=True, cache=True)
def tov_integrate(rho_c, rho_t, P_t, rec_r, rec_m, rec_P, rec_rho):
    """
    Integrate the TOV equations outward from the centre with an adaptive
    Dormand–Prince 5(4) scheme until P = P_STOP_FRAC · P_c.

    If the rec_* arrays have non-zero length, every accepted step is stored
    (r [cm], m [g], P [dyn/cm²], ρ [g/cm³]) - this is the radial profile.

    Returns (R [cm], M [g], number of recorded points).
    """
    Pc = interp_lin(rho_c, rho_t, P_t)
    P_stop = P_STOP_FRAC * Pc
    r = R_START
    m = FOUR_PI / 3.0 * r * r * r * rho_c
    P = Pc
    h = R_START

    n_cap = rec_r.shape[0]
    n_rec = 0
    if n_cap > 0:
        rec_r[0] = 0.0
        rec_m[0] = 0.0
        rec_P[0] = Pc
        rec_rho[0] = rho_c
        n_rec = 1

    k1m, k1P = tov_rhs(r, m, P, rho_t, P_t)
    for _ in range(MAX_STEPS):
        k2m, k2P = tov_rhs(r + _A21 * h, m + h * _A21 * k1m, P + h * _A21 * k1P, rho_t, P_t)
        k3m, k3P = tov_rhs(r + 0.3 * h,
                           m + h * (_A31 * k1m + _A32 * k2m),
                           P + h * (_A31 * k1P + _A32 * k2P), rho_t, P_t)
        k4m, k4P = tov_rhs(r + 0.8 * h,
                           m + h * (_A41 * k1m + _A42 * k2m + _A43 * k3m),
                           P + h * (_A41 * k1P + _A42 * k2P + _A43 * k3P), rho_t, P_t)
        k5m, k5P = tov_rhs(r + (8.0 / 9.0) * h,
                           m + h * (_A51 * k1m + _A52 * k2m + _A53 * k3m + _A54 * k4m),
                           P + h * (_A51 * k1P + _A52 * k2P + _A53 * k3P + _A54 * k4P), rho_t, P_t)
        k6m, k6P = tov_rhs(r + h,
                           m + h * (_A61 * k1m + _A62 * k2m + _A63 * k3m + _A64 * k4m + _A65 * k5m),
                           P + h * (_A61 * k1P + _A62 * k2P + _A63 * k3P + _A64 * k4P + _A65 * k5P),
                           rho_t, P_t)
        m_new = m + h * (_B1 * k1m + _B3 * k3m + _B4 * k4m + _B5 * k5m + _B6 * k6m)
        P_new = P + h * (_B1 * k1P + _B3 * k3P + _B4 * k4P + _B5 * k5P + _B6 * k6P)
        k7m, k7P = tov_rhs(r + h, m_new, P_new, rho_t, P_t)

        err_m = h * (_E1 * k1m + _E3 * k3m + _E4 * k4m + _E5 * k5m + _E6 * k6m + _E7 * k7m)
        err_P = h * (_E1 * k1P + _E3 * k3P + _E4 * k4P + _E5 * k5P + _E6 * k6P + _E7 * k7P)
        sc_m = RTOL * max(abs(m), abs(m_new))
        sc_P = RTOL * max(abs(P), abs(P_new)) + 1e-300
        err = max(abs(err_m) / sc_m, abs(err_P) / sc_P)

        if err <= 1.0:
            if P_new > P_stop:
                # accept
                r += h
                m = m_new
                P = P_new
                k1m, k1P = k7m, k7P
                if n_rec < n_cap:
                    rec_r[n_rec] = r
                    rec_m[n_rec] = m
                    rec_P[n_rec] = P
                    rec_rho[n_rec] = interp_lin(P, P_t, rho_t)
                    n_rec += 1
                fac = 5.0 if err < 1e-10 else min(5.0, 0.9 * err ** -0.2)
                h = min(h * fac, H_MAX)
            elif h > 1.0:
                # step overshoots the surface: shrink until it is < 1 cm
                h *= 0.5
            else:
                f = (P - P_stop) / (P - P_new)
                R = r + f * h
                M = m + f * (m_new - m)
                if n_rec < n_cap:
                    rec_r[n_rec] = R
                    rec_m[n_rec] = M
                    rec_P[n_rec] = P_stop
                    rec_rho[n_rec] = interp_lin(P_stop, P_t, rho_t)
                    n_rec += 1
                return R, M, n_rec
        else:
            h *= max(0.2, 0.9 * err ** -0.2)
    return r, m, n_rec


@njit(fastmath=True, cache=True)
def tov_mass_radius(rho_c, rho_t, P_t):
    """(R [cm], M [g]) for one central density, without recording a profile."""
    empty = np.empty(0)
    R, M, _ = tov_integrate(rho_c, rho_t, P_t, empty, empty, empty, empty)
    return R, M


@njit(parallel=True, fastmath=True, cache=True)
def solve_all_densities(rho_array, rho_eos, P_eos):
    """Solve for all central densities in parallel. Returns [R; M] in cgs."""
    n = rho_array.shape[0]
    results = np.zeros((2, n))
    for i in prange(n):
        R, M = tov_mass_radius(rho_array[i], rho_eos, P_eos)
        results[0, i] = R
        results[1, i] = M
    return results


@njit(fastmath=True, cache=True)
def tov_profile(rho_c, rho_t, P_t):
    """Radial profile arrays (r, m, P, ρ) in cgs for one central density."""
    rr = np.empty(PROFILE_MAX_POINTS)
    mm = np.empty(PROFILE_MAX_POINTS)
    pp = np.empty(PROFILE_MAX_POINTS)
    dd = np.empty(PROFILE_MAX_POINTS)
    _, _, n = tov_integrate(rho_c, rho_t, P_t, rr, mm, pp, dd)
    return rr[:n], mm[:n], pp[:n], dd[:n]


# ---------------------------------------------------------------------------
# Python-level computations (with caching)
# ---------------------------------------------------------------------------
def compute_eos(eos_name, rho_arr, P_arr):
    """Compute the M(ρc), R(ρc) sequence for one EOS."""
    with _solver_lock:
        results = solve_all_densities(RHO_C_GRID, rho_arr, P_arr)
    R = results[0] / 1e5      # km
    M = results[1] / M_SUN    # M☉
    i_max = int(np.argmax(M))
    return {
        'name': eos_name,
        'rho': RHO_C_GRID.tolist(),
        'R': R.tolist(),
        'M': M.tolist(),
        'max_mass': float(M[i_max]),
        'max_radius': float(R[i_max]),
        'rho_c_max': float(RHO_C_GRID[i_max]),
    }


def get_curve(eos_name):
    """Cached M–R curve for an EOS (computed once per EOS, not once per request)."""
    with _cache_lock:
        cached = _curve_cache.get(eos_name)
    if cached is not None:
        return cached
    eos = eos_cache[eos_name]
    curve = compute_eos(eos_name, eos['rho'], eos['P'])
    with _cache_lock:
        _curve_cache[eos_name] = curve
    return curve


def _brent(f, a, b, tol, maxiter=200):
    """Brent's method for a root of f in [a, b] (f(a), f(b) of opposite sign).
    Small stand-in for scipy.optimize.brentq, so scipy is not a dependency."""
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("Root is not bracketed")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:                       # keep the root between b and c
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):                 # b is the best estimate so far
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * np.finfo(float).eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:                        # secant step
                p, q = 2.0 * xm * s, 1.0 - s
            else:                             # inverse quadratic interpolation
                q, r = fa / fc, fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q               # accept interpolation
            else:
                d = e = xm                    # fall back to bisection
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    raise RuntimeError("Brent's method did not converge")


def find_rho_c_for_mass(target_mass_solar, eos_name):
    """Central density giving the target mass on the stable branch (ρc ≤ ρc(M_max))."""
    eos = eos_cache[eos_name]
    rho_t, P_t = eos['rho'], eos['P']
    curve = get_curve(eos_name)
    rho_grid = np.asarray(curve['rho'])
    M_grid = np.asarray(curve['M'])
    i_max = int(np.argmax(M_grid))
    M_max = M_grid[i_max]

    if target_mass_solar > M_max:
        raise ValueError(f"{target_mass_solar:.3f} M☉ exceeds the maximum mass "
                         f"{M_max:.3f} M☉ of this EOS")

    def mass_diff(rho_c):
        with _solver_lock:
            _, M = tov_mass_radius(rho_c, rho_t, P_t)
        return M / M_SUN - target_mass_solar

    stable_rho = rho_grid[:i_max + 1]
    stable_M = M_grid[:i_max + 1]

    if target_mass_solar < stable_M[0]:
        # light star: walk down in central density until the mass is bracketed
        rho_hi = stable_rho[0]
        rho_lo = rho_hi
        rho_floor = max(rho_t[0], 1e13)
        while True:
            rho_lo = rho_lo / 1.5
            if rho_lo < rho_floor:
                raise ValueError(f"{target_mass_solar:.3f} M☉ is below the lightest "
                                 f"star that can be bracketed for this EOS")
            if mass_diff(rho_lo) < 0:
                break
            rho_hi = rho_lo
    else:
        idx = int(np.searchsorted(stable_M, target_mass_solar))
        idx = min(max(idx, 1), len(stable_M) - 1)
        rho_lo, rho_hi = stable_rho[idx - 1], stable_rho[idx]
        if mass_diff(rho_lo) * mass_diff(rho_hi) > 0:
            # non-monotonic stretch of the curve: fall back to the full stable range
            rho_lo, rho_hi = stable_rho[0], stable_rho[-1]

    return _brent(mass_diff, rho_lo, rho_hi, tol=1e-10 * rho_hi)


def compute_profile(eos_name, mode, value):
    """
    Density profile ρ(r) of a single star.
    mode = 'mass'  -> value is the gravitational mass in M☉ (stable branch)
    mode = 'rho_c' -> value is the central density in g/cm³
    Returns (profile dict, list of warnings).
    """
    key = (eos_name, mode, float(value))
    with _cache_lock:
        cached = _profile_cache.get(key)
    if cached is not None:
        return cached

    warnings = []
    eos = eos_cache[eos_name]
    curve = get_curve(eos_name)

    if mode == 'mass':
        rho_c = find_rho_c_for_mass(value, eos_name)
    else:
        rho_c = float(value)
        if rho_c > eos['rho'][-1]:
            warnings.append(f"{eos_name}: ρc is above the last EOS table point "
                            f"({eos['rho'][-1]:.2e} g/cm³); the EOS is held constant there")
        if rho_c > curve['rho_c_max']:
            warnings.append(f"{eos_name}: ρc = {rho_c:.2e} g/cm³ is beyond the maximum-mass "
                            f"configuration (≈{curve['rho_c_max']:.2e} g/cm³) - unstable branch")

    with _solver_lock:
        r, m, P, rho = tov_profile(rho_c, eos['rho'], eos['P'])

    out = ({
        'rho_c': float(rho_c),
        'M': float(m[-1] / M_SUN),
        'R': float(r[-1] / 1e5),
        'r': (r / 1e5).tolist(),            # km
        'density': rho.tolist(),            # g/cm³
        'm': (m / M_SUN).tolist(),          # M☉
        'P': P.tolist(),                    # dyn/cm²
    }, warnings)
    with _cache_lock:
        if len(_profile_cache) >= PROFILE_CACHE_SIZE:
            _profile_cache.pop(next(iter(_profile_cache)))   # drop the oldest entry
        _profile_cache[key] = out
    return out


def compute_inertia(m, r):
    """Moment of inertia from an empirical fit, I = k(C)·M·R² (10⁴⁵ g·cm²)."""
    m = np.asarray(m, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    result = np.full_like(m, np.nan, dtype=np.float64)
    valid = (m > 0) & (r > 0) & np.isfinite(m) & np.isfinite(r)
    if not np.any(valid):
        return result
    m_v, r_v = m[valid], r[valid]
    compactness = np.clip(m_v * 1.477 / r_v, 0, 0.5)
    k = (0.237 + 0.008) * (1 + 4.2 * compactness + 90 * compactness ** 4)
    I = k * m_v * (r_v * 1e5) ** 2 * 1.989e33 / 1e45
    result[valid] = np.where(np.isfinite(I) & (I > 0), I, np.nan)
    return result


def _stable_inertia(result):
    """(M, I) on the stable branch, or None."""
    M = np.asarray(result.get('M', []), dtype=np.float64)
    R = np.asarray(result.get('R', []), dtype=np.float64)
    if len(M) == 0 or len(R) == 0:
        return None
    i_max = int(np.argmax(M))
    if i_max < 2:
        return None
    M_s, R_s = M[:i_max + 1], R[:i_max + 1]
    I = compute_inertia(M_s, R_s)
    ok = np.isfinite(I) & (I > 0) & (I < 1000)
    return M_s[ok], I[ok]


# ---------------------------------------------------------------------------
# Plotting (plain Plotly JSON; no server-side plotly import needed)
# ---------------------------------------------------------------------------
_AXIS_DARK = dict(gridcolor='#283442', linecolor='#506784', zerolinecolor='#283442',
                  ticks='', automargin=True)
_LAYOUT_DARK = dict(
    paper_bgcolor='rgb(17,17,17)', plot_bgcolor='rgb(17,17,17)',
    font=dict(color='#f2f5fa'), hovermode='closest',
    hoverlabel=dict(align='left'), legend=dict(bgcolor='rgba(0,0,0,0)'),
    margin=dict(l=70, r=30, t=60, b=60),
)


def _line(x, y, name, color, **extra):
    trace = dict(type='scatter', mode='lines', x=x, y=y, name=name,
                 line=dict(color=color or '#ffffff', width=2))
    trace.update(extra)
    return trace


def _layout(title, xaxis, yaxis):
    layout = dict(_LAYOUT_DARK)
    layout['title'] = dict(text=title)
    layout['xaxis'] = {**_AXIS_DARK, **xaxis}
    layout['yaxis'] = {**_AXIS_DARK, **yaxis}
    return layout


def create_plot(results, plot_type, options=None):
    """Return a Plotly figure dict {data, layout}."""
    options = options or {}
    data = []

    if plot_type == 'mass-rho':
        data = [_line(r['rho'], r['M'], r['name'], r.get('color')) for r in results]
        layout = _layout('Mass vs Central Density',
                         dict(title=dict(text='ρc (g/cm³)'), type='log'),
                         dict(title=dict(text='Mass (M☉)')))
    elif plot_type == 'radius-rho':
        data = [_line(r['rho'], r['R'], r['name'], r.get('color')) for r in results]
        layout = _layout('Radius vs Central Density',
                         dict(title=dict(text='ρc (g/cm³)'), type='log'),
                         dict(title=dict(text='Radius (km)'), range=[9, 30]))
    elif plot_type == 'mass-radius':
        data = [_line(r['R'], r['M'], r['name'], r.get('color')) for r in results]
        layout = _layout('Mass vs Radius',
                         dict(title=dict(text='Radius (km)'), range=[8, 20]),
                         dict(title=dict(text='Mass (M☉)'), range=[0, 3]))
    elif plot_type == 'inertia':
        for r in results:
            mi = _stable_inertia(r)
            if mi is not None and len(mi[0]):
                data.append(_line(mi[0].tolist(), mi[1].tolist(), r['name'], r.get('color')))
        layout = _layout('Moment of Inertia',
                         dict(title=dict(text='Mass (M☉)'), range=[1, 2.7]),
                         dict(title=dict(text='I (10⁴⁵ g·cm²)')))
    elif plot_type == 'eos':
        data = [_line(r['raw_rho'], r['raw_P'], r['name'], r.get('color'))
                for r in results if 'raw_rho' in r]
        layout = _layout('Equation of State: Pressure vs Density',
                         dict(title=dict(text='ρ (g/cm³)'), type='log'),
                         dict(title=dict(text='P (dyn/cm²)'), type='log'))
    elif plot_type == 'profile':
        log_y = bool(options.get('log_scale', False))
        for r in results:
            p = r.get('profile')
            if not p:
                continue
            label = f"{r['name']}  (M={p['M']:.3f} M☉, R={p['R']:.2f} km)"
            data.append(_line(
                p['r'], p['density'], label, r.get('color'),
                customdata=p['m'],
                hovertemplate=('r = %{x:.3f} km<br>ρ = %{y:.4e} g/cm³'
                               '<br>m(r) = %{customdata:.4f} M☉<extra>' + r['name'] + '</extra>')))
        layout = _layout('Density Profile ρ(r)',
                         dict(title=dict(text='r (km)'), rangemode='tozero'),
                         dict(title=dict(text='ρ (g/cm³)'),
                              type='log' if log_y else 'linear',
                              exponentformat='power'))
        if data:
            layout['shapes'] = [dict(type='line', xref='paper', x0=0, x1=1, yref='y',
                                     y0=RHO_SAT, y1=RHO_SAT,
                                     line=dict(color='#94a3b8', width=1, dash='dot'))]
            layout['annotations'] = [dict(xref='paper', x=1, xanchor='right', yref='y',
                                          y=np.log10(RHO_SAT) if log_y else RHO_SAT,
                                          yanchor='bottom', showarrow=False,
                                          text='ρ₀ (nuclear saturation)',
                                          font=dict(size=11, color='#94a3b8'))]
    else:
        layout = _layout(f'Invalid plot type: {plot_type}', {}, {})

    if not data:
        layout['annotations'] = [dict(text='No valid data to display', xref='paper',
                                      yref='paper', x=0.5, y=0.5, showarrow=False)]
    return {'data': data, 'layout': layout}


def generate_csv_data(results, plot_type):
    """Generate CSV data for the selected plot type"""
    if not results:
        return "No data available"

    lines = []
    if plot_type == 'mass-rho':
        lines.append("EOS,Central Density (g/cm³),Mass (M☉)")
        for r in results:
            lines += [f"{r['name']},{a},{b}" for a, b in zip(r.get('rho', []), r.get('M', []))]
    elif plot_type == 'radius-rho':
        lines.append("EOS,Central Density (g/cm³),Radius (km)")
        for r in results:
            lines += [f"{r['name']},{a},{b}" for a, b in zip(r.get('rho', []), r.get('R', []))]
    elif plot_type == 'mass-radius':
        lines.append("EOS,Radius (km),Mass (M☉)")
        for r in results:
            lines += [f"{r['name']},{a},{b}" for a, b in zip(r.get('R', []), r.get('M', []))]
    elif plot_type == 'inertia':
        lines.append("EOS,Mass (M☉),Inertia (10⁴⁵ g·cm²)")
        for r in results:
            mi = _stable_inertia(r)
            if mi is not None:
                lines += [f"{r['name']},{a},{b}" for a, b in zip(*mi)]
    elif plot_type == 'eos':
        lines.append("EOS,Density (g/cm³),Pressure (dyn/cm²)")
        for r in results:
            lines += [f"{r['name']},{a},{b}"
                      for a, b in zip(r.get('raw_rho', []), r.get('raw_P', []))]
    elif plot_type == 'profile':
        lines.append("EOS,Central Density (g/cm³),r (km),Density (g/cm³),"
                     "Enclosed Mass m(r) (M☉),Pressure (dyn/cm²)")
        for r in results:
            p = r.get('profile')
            if not p:
                continue
            rc = p['rho_c']
            lines += [f"{r['name']},{rc},{a},{b},{mm},{pp}"
                      for a, b, mm, pp in zip(p['r'], p['density'], p['m'], p['P'])]
    else:
        return f"Unknown plot type: {plot_type}"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Render main page"""
    eos_list = [{'name': name, 'color': color}
                for name, _, _, color in EOS_CONFIG if name in eos_cache]
    return render_template('index.html', eos_list=eos_list)


@app.route('/compute', methods=['POST'])
def compute():
    """Compute neutron star properties"""
    try:
        data = request.get_json(silent=True) or {}
        selected_eos = data.get('eos', [])
        plot_type = data.get('plot_type', 'mass-radius')
        profile_opts = data.get('profile') or {}

        if not selected_eos:
            return jsonify({'error': 'No EOS selected'}), 400

        mode = profile_opts.get('mode', 'mass')
        if plot_type == 'profile':
            if mode not in ('mass', 'rho_c'):
                return jsonify({'error': f'Unknown profile mode: {mode}'}), 400
            try:
                value = float(profile_opts.get('value'))
            except (TypeError, ValueError):
                return jsonify({'error': 'Enter a numeric mass or central density'}), 400
            if not np.isfinite(value) or value <= 0:
                return jsonify({'error': 'The mass / central density must be positive'}), 400

        results, warnings = [], []
        for eos_name in selected_eos:
            if eos_name not in eos_cache:
                warnings.append(f"{eos_name}: not loaded")
                continue
            eos = eos_cache[eos_name]
            curve = get_curve(eos_name)
            result = {k: curve[k] for k in ('name', 'rho', 'R', 'M', 'max_mass', 'max_radius')}
            result['color'] = eos['color']
            if plot_type == 'eos':
                result['raw_rho'] = eos['rho'].tolist()
                result['raw_P'] = eos['P'].tolist()
            elif plot_type == 'profile':
                try:
                    prof, w = compute_profile(eos_name, mode, value)
                    result['profile'] = prof
                    warnings += w
                except ValueError as e:
                    warnings.append(f"{eos_name}: {e}")
                    continue
            results.append(result)

        options = {'log_scale': bool(profile_opts.get('log_scale', False))}
        return jsonify({
            'plot': create_plot(results, plot_type, options),
            'results': results,
            'warnings': warnings,
        })

    except Exception as e:
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
        rho, P = _clean_eos(*_read_rho_P(file.read().decode('utf-8-sig')))

        eos_name = secure_filename(file.filename).replace('.csv', '').upper() or 'CUSTOM'
        with _cache_lock:
            eos_cache[eos_name] = {'rho': rho, 'P': P, 'color': '#FF0000'}
            _curve_cache.pop(eos_name, None)   # invalidate any old results
            for k in [k for k in _profile_cache if k[0] == eos_name]:
                del _profile_cache[k]
        return jsonify({'success': True, 'name': eos_name, 'color': '#FF0000'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Download CSV with the calculated data of the selected plot type"""
    try:
        data = request.get_json(silent=True) or {}
        results = data.get('results', [])
        plot_type = data.get('plot_type', 'mass-radius')
        if not results:
            return jsonify({'error': 'No data available for download'}), 400
        csv_data = generate_csv_data(results, plot_type)
        return send_file(
            io.BytesIO(csv_data.encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'neutron_star_{plot_type}_data.csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Start-up: load EOS tables and compile the Numba kernels once.
# Runs at import time so it also works under gunicorn.
# ---------------------------------------------------------------------------
def _warmup():
    if not eos_cache:
        return
    eos = next(iter(eos_cache.values()))
    try:
        solve_all_densities(RHO_C_GRID[:4], eos['rho'], eos['P'])
        tov_profile(5e14, eos['rho'], eos['P'])
        tov_mass_radius(5e14, eos['rho'], eos['P'])
    except Exception as e:  # never block start-up; kernels compile on first use instead
        print(f"Warm-up skipped: {e}")


load_eos_data()
_warmup()

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    print(f"\nApplication ready! Starting Flask server on port {port} (debug={debug})...")
    app.run(debug=debug, host='0.0.0.0', port=port)
