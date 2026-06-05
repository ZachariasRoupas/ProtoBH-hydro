#!/usr/bin/env python3
"""
================================================================================
2D BH Accretion Simulation Analysis — Simple Harmonic Orbit
Part of the project: protoBH — MSCA grant agreement No. 101149270 funded by the E.U.
Author: Zacharias Roupas
Please cite:
  Roupas, Z., to appear (2026)
  Code DOI: https://doi.org/10.5281/zenodo.XXXXXXX
Built on Athena++ v21.0 (Stone et al. 2020, ApJS 249, 4)
================================================================================
Unit system (set in C++ problem generator):
  length_unit  = R_Bondi_max = 2 G m_BH / (c_s^2 * (1 + Mach_min^2))
  velocity_unit = c_s
  time_unit    = R_Bondi_max / c_s
  density_unit = rho_gas = (1 - epsilon) * rho_cluster
"""

# Disclaimer: This software is provided "as is", without warranty of any kind.
# The author accepts no liability for any errors or consequences of its use.
# Users are responsible for verifying results against the cited publication.

import sys
import json
import numpy as np
from pathlib import Path
import athena_read as ar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# TEE — mirror stdout to a log file
# ============================================================

class _Tee:
    """Write to both the real stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._stdout = sys.stdout
        self._log    = open(log_path, 'w', buffering=1)
        sys.stdout   = self
    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stdout.flush()
        self._log.flush()
    def close(self):
        sys.stdout = self._stdout
        self._log.close()

# ============================================================
# CONFIGURATION 
# ============================================================
DIR_NAME     = "mBH50.0_B0.010_ecc0.00_psi00.00_nR168_nP384_rho1.0e+07/nu0e+00_NonI1_Rinn5.0e-03_Rout2.00_tend8.00"
SIM_DATA_DIR = Path(f"/simulation_2D/{DIR_NAME}")
OUTPUT_DIR   = Path(f"/images_2D/{DIR_NAME}")

BETA_THRESHOLD = 1.0
VR_THRESHOLD = 0.0

# Index of the snapshot to use for all "final time" radial profile plots.
# Set to None to use the last available snapshot automatically.
FINAL_SNAP_IDX = None

matplotlib.rcParams.update({
    'text.usetex':          True,       # LaTeX rendering (requires a TeX installation)
    'font.family':          'serif',
    'axes.labelsize':       24,
    'xtick.labelsize':      16,
    'ytick.labelsize':      16,
    'legend.fontsize':      12,
    'xtick.major.size':     6,
    'ytick.major.size':     6,
    'xtick.minor.size':     4,
    'ytick.minor.size':     4,
    'xtick.major.width':    1.0,
    'ytick.major.width':    1.0,
    'xtick.top':            True,
    'ytick.right':          True,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'lines.linewidth':      2.0,
    'axes.grid':            False,
    'figure.figsize':   (8, 5),   # inches
    'savefig.dpi':      300,      # resolution when saving to file
})

# ============================================================
# I/O
# ============================================================

def read_parameters(sim_dir):
    with open(sim_dir / "parameters.json") as f:
        return json.load(f)


def read_snapshots(sim_dir):
    prim_files = sorted(sim_dir.glob("*.prim.*.athdf"))
    if not prim_files:
        raise FileNotFoundError(f"No prim files in {sim_dir}")
    prim_list = [ar.athdf(str(f)) for f in prim_files]
    return prim_list, prim_files

# ============================================================
# DATA EXTRACTION
# ============================================================

def orbital_period(params):
    """Orbital period in code units: T = 2π / Omega_code."""
    return 2.0 * np.pi / params['code_units']['Omega_code']


def phi_average(snap, field):
    """Azimuthal average of a field for a single snapshot. Returns shape (n_r,)."""
    return np.mean(np.squeeze(snap[field]), axis=0)


def extract_timeseries(prim_list, x1v, params):
    """
    Loop over snapshots and extract:
      t_per    : time in orbital periods,          shape (n_snap,)
      vphi_avg : phi-averaged v_phi at all radii,  shape (n_snap, n_r)
      vr_avg   : phi-averaged v_r   at all radii,  shape (n_snap, n_r)
      beta_prof: β = <v_phi>^2/v_Kep^2 profile,   shape (n_snap, n_r)
                 non-positive and non-finite values replaced with NaN
                 so that log-scale plots never crash.
      v_kep    : Kepler velocity
    """
    gm     = params['code_units']['gm_code']
    T_orb  = orbital_period(params)
    v_kep  = np.sqrt(gm / x1v)          # shape (n_r,), computed once

    n_snap = len(prim_list)
    n_r    = len(x1v)

    t_per     = np.zeros(n_snap)
    vphi_avg  = np.zeros((n_snap, n_r))
    vr_avg    = np.zeros((n_snap, n_r))
    beta_prof = np.zeros((n_snap, n_r))

    for k, snap in enumerate(prim_list):
        t_per[k]      = snap['Time'] / T_orb
        vphi_avg[k]   = phi_average(snap, 'vel2')
        vr_avg[k]     = phi_average(snap, 'vel1')
        beta_prof[k]  = (vphi_avg[k] / v_kep) ** 2

    # Replace non-finite and non-positive values with NaN.
    # This keeps retrograde (<v_phi> < 0) snapshots visible in linear plots
    # while preventing log-scale crashes.
    beta_prof = np.where(np.isfinite(beta_prof) & (beta_prof > 0.0),
                         beta_prof, np.nan)

    return t_per, vphi_avg, vr_avg, beta_prof, v_kep


# ============================================================
# DISK DETECTION
# ============================================================

def detect_disk(vphi_avg_kep, vr_avg_ff, x1v, t_per, idx_inner, idx_outer, threshold_beta, threshold_vr, it_last):
    """
    Find the earliest time and outermost radius where beta > threshold_beta.
    Returns (disk_t_per, disk_r, disk_ir, disk_k) or (None, None, None, None).
    """
    t_out   = None
    x1v_out = None
    ir_out  = None
    k_out   = None

    # =================================================
    # === Detect the disk radius and formation time ===

    beta_bc = vphi_avg_kep[0, idx_outer] * vphi_avg_kep[0, idx_outer]

    # identify smaller r where thresholds are not met by ic     
    ir_bc = idx_outer
    bc_found = False
    if beta_bc > threshold_beta:
        ir = idx_outer
        while ir > idx_inner and not bc_found:
            vp = vphi_avg_kep[it_last, ir]
            beta = vp * vp
            if beta < threshold_beta: 
                ir_bc = ir
                bc_found = True
            ir -= 1

    # identify disk radius and time when disk formation is completed     
    disk_found = False
    for k in range(0, it_last + 1):
        for ir in range(idx_inner,ir_bc + 1):
            vr = vr_avg_ff[k, ir]
            beta = vphi_avg_kep[k, ir] * vphi_avg_kep[k, ir]
            if beta > threshold_beta and vr > threshold_vr:
                if not disk_found:
                    x1v_out = x1v[ir]
                    ir_out = ir
                    t_out   = t_per[k]
                    k_out   = k
                    disk_found = True
                else:
                    if x1v_out is not None and vr > threshold_vr:
                        if x1v[ir] > x1v_out:
                            x1v_out = x1v[ir]
                            ir_out = ir
                            t_out   = t_per[k]
                            k_out   = k                            
        
    return t_out, x1v_out, ir_out, k_out


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary(params, prim_list, prim_files, t_per,
                  disk_t_per, disk_r, T_orb_Myr, T_orb_yr,
                  R_circ_code, R_circ_pc, tau_prop_Porb, tau_prop_yr):
    pi  = params['physical_input']
    dq  = params['derived_quantities_cgs']
    us  = params['unit_system_cgs']
    cu  = params['code_units']
    cst = params['physical_constants_cgs']

    print("\n" + "="*60)
    print("SIMULATION  —  ORBITING BH  (Elliptical Orbit)")
    print("="*60)
    print(f"  M_BH           = {pi['M_BH_Msun']:.1f} M_sun")
    print(f"  B (minor)      = {pi['B_pc']:.4e} pc"
          f"  |  A (major) = {dq['A_pc']:.4e} pc")
    print(f"  eccentricity   = {pi['ecc']:.4e}")
    print(f"  rho_gas        = {pi['rho_gas_cgs']:.3e} g/cm^3")
    print(f"  T_gas          = {pi['T_gas_K']:.1f} K")
    print(f"  c_s            = {dq['c_s_cgs']*1e-5:.4e} km/s")
    print(f"  Orbital period = {T_orb_Myr:.4e} Myr  ({T_orb_yr:.4e} yr)")
    print(f"  length_unit    = {us['length_unit_cgs']/cst['au_cgs']:.4e} AU"
          f"  = {us['length_unit_cgs']/cst['pc_cgs']:.4e} pc")
    print(f"  time_unit      = {us['time_unit_yr']:.4e} yr")
    print(f"  gm_code        = {cu['gm_code']:.4e}")
    print(f"  Omega_code     = {cu['Omega_code']:.4e}")
    print(f"  Snapshots      = {len(prim_list)}"
          f"  ({prim_files[0].name} … {prim_files[-1].name})")
    print(f"  t_end          = {prim_list[-1]['Time']:.4e} code"
          f"  = {t_per[-1]:.4e} periods")
    print(f"\n  beta threshold     = {BETA_THRESHOLD}")
    print(f"  v_r/v_ff threshold = {VR_THRESHOLD}")
    print(f"\n  R_circ             = {R_circ_code:.4e} R_Bondi"
          f"  =  {R_circ_pc:.4e} pc")
    print(f"  tau_prop           = {tau_prop_Porb:.4e} P_orb"
          f"  =  {tau_prop_yr:.4e} yr")
    if disk_t_per is not None:
        print(f"  Disk formation completed at"
              f"  t = {disk_t_per:.4e} T_orb,"
              f"  r = {disk_r:.4e} length_unit")
    else:
        print( "  No disk detected everywhere)")
    print("="*60 + "\n")


# ============================================================
# HELPERS
# ============================================================

def save_fig(fig, name):
    fig.savefig(OUTPUT_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(which='both', top=True, right=True, direction='in')

def safe_log_ylim(data, margin=2.0, floor=1.0e-6):
    """
    Return (ymin, ymax) for a log-scale axis from finite positive data.
    Returns None if no valid data exist.
    """
    finite = data[np.isfinite(data) & (data > 0.0)]
    if finite.size == 0:
        return None
    return max(floor, finite.min() / margin), finite.max() * margin

def gen_radial_ind(x1v):
    n_r          = len(x1v)
    idx_inner    = 0            # one cell in from inner BC
    idx_outer    = n_r - 1      # one cell in from outer BC
    idx_median   = (idx_inner + idx_outer) // 2
    return idx_inner, idx_median, idx_outer

# ============================================================
# INDIVIDUAL PLOT FUNCTIONS
# ============================================================

def plot_vphi_over_vkep(t_per, vphi_avg, x1v, idx_inner, params, disk_ir, disk_k):
    gm      = params['code_units']['gm_code']
    r       = x1v[idx_inner]
    v_kep   = np.sqrt(gm / r)

    fig, ax = plt.subplots()
    ax.plot(t_per, vphi_avg[:, idx_inner] / v_kep, 'b-', lw=1.8)
    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if disk_k is not None:
        ax.axvline(t_per[disk_k], color='k', ls='--', lw=0.8, alpha=0.5)
        vphi_steady = -1.0
        if vphi_avg[disk_k, disk_ir] > 0:
            vphi_steady = 1.0
        ax.axhline(vphi_steady, color='k', ls='--', lw=0.8, alpha=0.5)
    style_ax(ax, r'$t / P_{\rm orb}$', r'$\langle v_\phi \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, f'vphi_over_vkep_inner_r{r:.4f}.png')

def plot_vr(t_per, vr_avg, x1v, idx_inner, disk_t):
    r       = x1v[idx_inner]
    fig, ax = plt.subplots()
    ax.plot(t_per, vr_avg[:, idx_inner], 'g-', lw=1.8)
    ax.axhline(0, color='k', ls='--', lw=0.8, alpha=0.5)
    if disk_t is not None:
        ax.axvline(disk_t, color='k', ls='--', lw=0.8, alpha=0.5)
    style_ax(ax, r'$t / P_{\rm orb}$', r'$\langle v_r \rangle_\phi / c_s$')
    plt.tight_layout()
    save_fig(fig, f'vr_inner_r{r:.4f}.png')


def plot_beta_timeseries(t_per, beta_prof, x1v, disk_t_per, idx_inner, idx_median, idx_outer):
    configs = [
        (idx_inner,  'inner',  'b'),
        (idx_median, 'median', 'r'),
        (idx_outer,  'outer',  'darkorange'),
    ]
    for idx, label, color in configs:
        r        = x1v[idx]
        series   = beta_prof[:, idx]
        fig, ax  = plt.subplots()
        ax.plot(t_per, series, color=color, lw=1.8)
        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_yscale('log')
        lim = safe_log_ylim(series)
        if disk_t_per is not None:
            ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)
        if lim is not None:
            ax.set_ylim(*lim)
        else:
            # No positive data — fall back to linear scale with a warning
            ax.set_yscale('linear')
            print(f"  WARNING: beta_{label} has no positive values — " f"using linear scale")
        style_ax(ax, r'$t / P_{\rm orb}$', r'$\beta = \langle v_\phi \rangle_\phi^2 / v_{\rm Kep}^2$')
        plt.tight_layout()
        save_fig(fig, f'beta_r{r:.4f}_{label}.png')


def plot_beta_profile(t_per, beta_prof, x1v, k, label, disk_r):
    profile = beta_prof[k, :]
    lim     = safe_log_ylim(profile)
    if lim is None:
        print(f"  WARNING: skipping beta_profile '{label}' — "
              f"no positive β values at snapshot k={k}")
        return
    fig, ax = plt.subplots()
    ax.plot(x1v, profile, 'b-', lw=1.8)
    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if disk_r is not None:
        ax.axvline(disk_r, color='b', ls=':', lw=1.0)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(*lim)
    style_ax(ax, r'$r/R_{\rm B}$', r'$\beta = \langle v_\phi \rangle_\phi^2 / v_{\rm Kep}^2$')
    plt.tight_layout()
    save_fig(fig, f'beta_profile_t{t_per[k]:.4f}_{label.replace(" ", "_")}.png')

def plot_profile_disk(t_per, vphi_avg, vr_avg, x1v, disk_t_per, disk_r, disk_ir, disk_k):
    vphi_t = vphi_avg[:, disk_ir]
    vphi_r = vphi_avg[disk_k, :]
    vr_t = vr_avg[:, disk_ir]
    vr_r = vr_avg[disk_k, :]

    vphi_steady = -1.0
    if vphi_t[disk_k] > 0:
        vphi_steady = 1.0

    fig1, ax1 = plt.subplots()
    ax1.plot(x1v, vphi_r, 'b-', lw=1.8)
    ax1.set_xscale('log')
    ax1.axhline(0.0, color='b', ls=':', lw=1.0)
    ax1.axhline(vphi_steady, color='b', ls=':', lw=1.0)
    ax1.axvline(disk_r, color='b', ls=':', lw=1.0)
    style_ax(ax1, r'$r/R_{\rm B}$', r'$\langle v_\phi(t_{\rm d},r) \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig1, f'disk_detection_vp_r_t{disk_t_per:.4f}.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(t_per, vphi_t, 'b-', lw=1.8)
    ax2.axhline(0.0, color='b', ls=':', lw=1.0)
    ax2.axhline(vphi_steady, color='b', ls=':', lw=1.0)
    ax2.axvline(disk_t_per, color='b', ls=':', lw=1.0)
    style_ax(ax2, r'$t / P_{\rm orb}$', r'$\langle v_\phi(t,r_{\rm d}) \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig2, f'disk_detection_vp_t_r{disk_r:.4f}.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(x1v, vr_r, 'b-', lw=1.8)
    ax3.set_xscale('log')
    ax3.axhline(0.0, color='b', ls=':', lw=1.0)
    ax3.axvline(disk_r, color='b', ls=':', lw=1.0)
    style_ax(ax3, r'$r/R_{\rm B}$', r'$\langle v_r(t_{\rm d},r) \rangle_\phi / c_s$')
    plt.tight_layout()
    save_fig(fig3, f'disk_detection_vr_r_t{disk_t_per:.4f}.png')

    fig4, ax4 = plt.subplots()
    ax4.plot(t_per, vr_t, 'b-', lw=1.8)
    ax4.axhline(0.0, color='b', ls=':', lw=1.0)
    ax4.axvline(disk_t_per, color='b', ls=':', lw=1.0)
    style_ax(ax4, r'$t / P_{\rm orb}$', r'$\langle v_r(t,r_{\rm d}) \rangle_\phi / c_s$')
    plt.tight_layout()
    save_fig(fig4, f'disk_detection_vr_t_r{disk_r:.4f}.png')

def plot_beta_overlay(t_per, beta_prof, x1v, idx_inner, disk_ir, disk_r, disk_t_per):
    """
    Overlay of beta = <v_phi>^2/v_Kep^2 vs time at five radii.
    Only called when a disk is detected.
    """
    r_inner = x1v[idx_inner]
    r_disk  = x1v[disk_ir]
    r_1   = 0.5 * (r_inner + r_disk)
    idx_1 = np.argmin(np.abs(x1v - r_1))
    r_1   = x1v[idx_1]
    r_2   = 5.0 * r_disk
    idx_2 = np.argmin(np.abs(x1v - r_2))
    r_2   = x1v[idx_2]
    r_3   = 10.0 * r_disk
    idx_3 = np.argmin(np.abs(x1v - r_3))
    r_3   = x1v[idx_3]

    fig, ax = plt.subplots()
    ax.plot(t_per, beta_prof[:, idx_inner],
            color='m', ls='--', lw=1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, beta_prof[:, idx_1],
            color='r', ls='-.', lw=1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, beta_prof[:, disk_ir],
            color='b', ls='-',           label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, beta_prof[:, idx_2],
            color='darkorange', ls=':', lw=1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, beta_prof[:, idx_3],
            color='k', ls=':', lw=1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_3)

    ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)

    ax.set_yscale('log')
    all_series = beta_prof[:, [idx_inner, idx_1, disk_ir, idx_2, idx_3]]
    lim = safe_log_ylim(all_series)
    if lim is not None:
        ax.set_ylim(*lim)
    else:
        ax.set_yscale('linear')
        print("  WARNING: beta_overlay has no positive values — using linear scale")

    ax.legend(frameon=False)
    style_ax(ax, r'$t / P_{\rm orb}$',
             r'$\beta = \langle v_\phi \rangle_\phi^2 / v_{\rm Kep}^2$')
    plt.tight_layout()
    save_fig(fig, 'beta_vs_t_multiradius.png')

def plot_vphi_kep_overlay(t_per, vphi_avg_kep, x1v, idx_inner, disk_ir, disk_r, disk_t_per):
    """
    Overlay of <v_phi>/v_Kep vs time at the inner boundary (dashed red),
    the detected disk radius (solid blue), and the midpoint between
    r_disk and r_outer (solid green).
    Only called when a disk is detected.
    """
    r_inner = x1v[idx_inner]
    r_disk  = x1v[disk_ir]

    r_1   = 0.5 * (r_inner + r_disk)
    idx_1 = np.argmin(np.abs(x1v - r_1))
    r_1   = x1v[idx_1]
    r_2   = 5.0 * r_disk
    idx_2 = np.argmin(np.abs(x1v - r_2))
    r_2   = x1v[idx_2]
    r_3   = 10.0 * r_disk
    idx_3 = np.argmin(np.abs(x1v - r_3))
    r_3   = x1v[idx_3]

    fig, ax = plt.subplots()
    ax.plot(t_per, vphi_avg_kep[:, idx_inner],
            color='m', ls='--', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, vphi_avg_kep[:, idx_1],
            color='r', ls='-.', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, vphi_avg_kep[:, disk_ir],
            color='b', ls='-', label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, vphi_avg_kep[:, idx_2],
            color='darkorange', ls=':', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, vphi_avg_kep[:, idx_3],
            color='k', ls=':', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_3)

    vmax = np.nanmax(vphi_avg_kep[:, [idx_inner, idx_1, disk_ir, idx_2, idx_3]])

    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmax > 1.0:
        ax.axhline( 1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)

    ax.legend(frameon=False)
    style_ax(ax, r'$t /P_{\rm orb}$', r'$\langle v_\phi \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()

    save_fig(fig, 'vphi_vs_t_multiradius.png')

def plot_vr_ff_overlay(t_per, vr_avg_ff, x1v, idx_inner, disk_ir, disk_t_per):
    """
    Overlay of <v_r>/v_ff vs time at the inner boundary (dashed red)
    and at the detected disk radius (solid blue).
    Only called when a disk is detected.
    """
    r_inner = x1v[idx_inner]
    r_disk  = x1v[disk_ir]

    r_1   = 0.5 * (r_inner + r_disk)
    idx_1 = np.argmin(np.abs(x1v - r_1))
    r_1   = x1v[idx_1]
    r_2   = 5.0 * r_disk
    idx_2 = np.argmin(np.abs(x1v - r_2))
    r_2   = x1v[idx_2]
    r_3   = 10.0 * r_disk
    idx_3 = np.argmin(np.abs(x1v - r_3))
    r_3   = x1v[idx_3]

    fig, ax = plt.subplots()
    ax.plot(t_per, vr_avg_ff[:, idx_inner],
            color='m', ls='--', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_inner)
    ax.plot(t_per, vr_avg_ff[:, idx_1],
            color='r', ls='-.', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_1)
    ax.plot(t_per, vr_avg_ff[:, disk_ir],
            color='b', ls='-',  label=r'$r = %.4f \; R_{\rm B}$' % r_disk)
    ax.plot(t_per, vr_avg_ff[:, idx_2],
            color='darkorange', ls=':', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_2)
    ax.plot(t_per, vr_avg_ff[:, idx_3],
            color='k', ls=':', lw = 1.0, label=r'$r = %.4f \; R_{\rm B}$' % r_3)

    vmax = np.nanmax(vr_avg_ff[:, [idx_inner, idx_1, disk_ir, idx_2, idx_3]])

    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    if vmax > 1.0:
        ax.axhline( 1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    ax.axvline(disk_t_per, color='k', ls='--', lw=0.8, alpha=0.5)

    leg = ax.legend(frameon=False)
    style_ax(ax, r'$t / P_{\rm orb}$',
             r'$\langle v_r \rangle_\phi / v_{\rm ff}(r)$')
    plt.tight_layout()
    save_fig(fig, 'vr_vs_t_multiradius.png')


def plot_vphi_kep_multitime(t_per, vphi_avg_kep, x1v, disk_k, disk_r, k_final):
    """
    Radial profiles of <v_phi>/v_Kep at t=0, disk detection time, and final time.
    Only called when a disk is detected.
    """

    k_half = np.argmin(np.abs(t_per - t_per[disk_k] / 2.0))

    configs = [
        (0,       'k',          '--', r'$t = 0$'),
        (k_half,  'g',          ':',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_half]),
        (disk_k,  'b',          '-',  r'$t = %.3f \; P_{\rm orb}$' % t_per[disk_k]),
        (k_final, 'r',          '-.',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_final]),
    ]

    # reference line guard: check across all three snapshots
    indices = [0, k_half, disk_k, k_final]
    vmax = np.nanmax(vphi_avg_kep[np.ix_(indices, range(len(x1v)))])
    vmin = np.nanmin(vphi_avg_kep[np.ix_(indices, range(len(x1v)))])

    fig, ax = plt.subplots()
    for k, color, ls, lbl in configs:
        ax.plot(x1v, vphi_avg_kep[k, :], color=color, ls=ls, label=lbl)

    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmax > 1.0:
        ax.axhline( 1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if vmin < -1.0:
        ax.axhline(-1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if disk_r is not None:
        ax.axvline(disk_r, color='gray', ls=':', lw=0.8, alpha=0.5)

    ax.set_xscale('log')
    ax.legend(frameon=False)
    style_ax(ax, r'$r / R_{\rm B}$',
             r'$\langle v_\phi \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, 'vphi_vs_r_multitime.png')


def plot_vr_ff_multitime(t_per, vr_avg_ff, x1v, disk_k, disk_r, k_final):
    """
    Radial profiles of <v_r>/v_ff at t=0, disk detection time, and final time.
    Only called when a disk is detected.
    """

    k_half = np.argmin(np.abs(t_per - t_per[disk_k] / 2.0))

    configs = [
        (k_half,  'g',          ':',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_half]),
        (disk_k,  'b',          '-',  r'$t = %.3f \; P_{\rm orb}$' % t_per[disk_k]),
        (k_final, 'r',          '-.',  r'$t = %.3f \; P_{\rm orb}$' % t_per[k_final]),
    ]

    indices = [disk_k, k_final]
    vmax = np.nanmax(vr_avg_ff[np.ix_(indices, range(len(x1v)))])
    vmin = np.nanmin(vr_avg_ff[np.ix_(indices, range(len(x1v)))])

    fig, ax = plt.subplots()
    for k, color, ls, lbl in configs:
        ax.plot(x1v, vr_avg_ff[k, :], color=color, ls=ls, label=lbl)

    ax.axhline(0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    if vmin < -1.0:
        ax.axhline(-1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if vmax > 1.0:
        ax.axhline( 1.0, color='k', ls=':', lw=0.8, alpha=0.5)
    if disk_r is not None:
        ax.axvline(disk_r, color='gray', ls=':', lw=0.8, alpha=0.5)

    ax.set_xscale('log')
    ax.legend(frameon=False)
    style_ax(ax, r'$r / R_{\rm B}$',
             r'$\langle v_r \rangle_\phi / v_{\rm ff}(r)$')
    plt.tight_layout()
    save_fig(fig, 'vr_vs_r_multitime.png')

def plot_vphi_kep_profile(t_per, vphi_avg_kep, x1v, k, label, disk_r=None):
    """
    Radial profile of <v_phi>/v_Kep at snapshot index k.
    Always produced, independent of disk detection.
    disk_r: if not None, marks the detected disk radius with a dotted vertical line.
    """
    profile = vphi_avg_kep[k, :]
    fig, ax = plt.subplots()
    ax.plot(x1v, profile, 'b-', lw=1.8)
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline( 1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5)
    if disk_r is not None:
        ax.axvline(disk_r, color='b', ls=':', lw=1.0, label=r'$r_{\rm disk}$')
        ax.legend()
    ax.set_xscale('log')
    style_ax(ax, r'$r/R_{\rm B}$',
             r'$\langle v_\phi \rangle_\phi / v_{\rm Kep}$')
    plt.tight_layout()
    save_fig(fig, f'vphi_kep_profile_{label.replace(" ", "_")}.png')


def plot_vr_freefall_profile(t_per, vr_avg, x1v, params, k, label, disk_r=None):
    """
    Radial profile of <v_r>/v_ff(r) at snapshot index k,
    where v_ff(r) = sqrt(2*GM/r) is the local free-fall speed.
    Always produced, independent of disk detection.
    disk_r: if not None, marks the detected disk radius with a dotted vertical line.
    """
    gm    = params['code_units']['gm_code']
    v_ff  = np.sqrt(2.0 * gm / x1v)    # shape (n_r,)
    ratio = vr_avg[k, :] / v_ff

    fig, ax = plt.subplots()
    ax.plot(x1v, ratio, 'g-', lw=1.8)
    ax.axhline( 0.0, color='k', ls='--', lw=0.8, alpha=0.5)
    ax.axhline(-1.0, color='k', ls=':',  lw=0.8, alpha=0.5,
               label=r'$v_r = -v_{\rm ff}$')
    if disk_r is not None:
        ax.axvline(disk_r, color='g', ls=':', lw=1.0, label=r'$r_{\rm disk}$')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    style_ax(ax, r'$r/R_{\rm B}$',
             r'$\langle v_r \rangle_\phi / v_{\rm ff}(r)$')
    plt.tight_layout()
    save_fig(fig, f'vr_freefall_profile_{label.replace(" ", "_")}.png')


def plot_vphi_radial_envelope(t_per, vphi_avg, x1v, params):
    """
    Temporal variation amplitude of <v_phi>/v_Kep as a function of radius.
    Uses nanmax/nanmin to ignore NaN entries safely.
    """
    gm    = params['code_units']['gm_code']
    v_kep = np.sqrt(gm / x1v)
    ratio = vphi_avg / v_kep[np.newaxis, :]        # shape (n_snap, n_r)

    var_amplitude = np.nanmax(ratio, axis=0) - np.nanmin(ratio, axis=0)

    fig, ax = plt.subplots()
    ax.plot(x1v, var_amplitude, 'b-', lw=1.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    style_ax(ax, r'$r/R_{\rm B}$', r'$\Delta(\langle v_\phi\rangle / v_{\rm Kep})$')
    ax.set_title('Temporal variation amplitude vs radius', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'vphi_variation_amplitude.png')


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _log = _Tee(OUTPUT_DIR / "run_log.txt")

    params                = read_parameters(SIM_DATA_DIR)
    prim_list, prim_files = read_snapshots(SIM_DATA_DIR)

    x1v                              = prim_list[0]['x1v']
    idx_inner, idx_median, idx_outer = gen_radial_ind(x1v)

    gm = params['code_units']['gm_code']

    t_per, vphi_avg, vr_avg, beta_prof, v_kep = extract_timeseries(prim_list, x1v, params)
    vphi_avg_kep = vphi_avg / v_kep
    v_ff      = np.sqrt(2.0 * gm / x1v)                 # shape (n_r,)
    vr_avg_ff = vr_avg / v_ff[np.newaxis, :]             # shape (n_snap, n_r)

    k_final = FINAL_SNAP_IDX if FINAL_SNAP_IDX is not None else len(prim_list) - 1

    disk_t_per, disk_r, disk_ir, disk_k = detect_disk(vphi_avg_kep, vr_avg_ff, x1v, t_per, idx_inner, idx_outer, 
                                                        BETA_THRESHOLD, VR_THRESHOLD, k_final)

    print(f"  Final-time profiles use snapshot k={k_final}"
          f"  (t = {t_per[k_final]:.4e} T_orb)")

    T_orb     = orbital_period(params)
    T_orb_yr  = T_orb * params['unit_system_cgs']['time_unit_yr']
    T_orb_Myr = T_orb_yr / 1.0e6

    # R_circ = Omega^2 * R_Bondi^4 / (G*M_BH)  with R_Bondi = 1 (code length unit)
    # tau_prop = sqrt(R_circ^3 / (G*M_BH)) 
    Omega_code    = params['code_units']['Omega_code']
    gm_code       = params['code_units']['gm_code']
    R_circ_code   = Omega_code**2 / gm_code                    # [R_Bondi]
    tau_prop_code = np.sqrt(R_circ_code**3 / gm_code)          # [time_unit]
    tau_prop_yr   = tau_prop_code * params['unit_system_cgs']['time_unit_yr']
    tau_prop_Porb = tau_prop_code / T_orb
    length_unit_pc = (params['unit_system_cgs']['length_unit_cgs']
                      / params['physical_constants_cgs']['pc_cgs'])
    R_circ_pc     = R_circ_code * length_unit_pc                # [pc]

    print_summary(params, prim_list, prim_files, t_per,
                  disk_t_per, disk_r, T_orb_Myr, T_orb_yr,
                  R_circ_code, R_circ_pc, tau_prop_Porb, tau_prop_yr)

    plot_vphi_over_vkep(t_per, vphi_avg, x1v, idx_inner, params, disk_ir, disk_k)
    plot_vr(t_per, vr_avg, x1v, idx_inner, disk_t_per)
    plot_beta_timeseries(t_per, beta_prof, x1v, disk_t_per, idx_inner, idx_median, idx_outer)
    plot_beta_profile(t_per, beta_prof, x1v, k_final, 'final time', disk_r)
    plot_vphi_kep_profile(t_per, vphi_avg_kep, x1v, k_final, 'final time', disk_r)
    plot_vr_freefall_profile(t_per, vr_avg, x1v, params, k_final, 'final time', disk_r)
    plot_vphi_radial_envelope(t_per, vphi_avg, x1v, params)

    if disk_k is not None:
        plot_profile_disk(t_per, vphi_avg_kep, vr_avg, x1v, disk_t_per, disk_r, disk_ir, disk_k)
        plot_beta_overlay(t_per, beta_prof, x1v, idx_inner, disk_ir, disk_r, disk_t_per)
        plot_vphi_kep_overlay(t_per, vphi_avg_kep, x1v, idx_inner, disk_ir, disk_r, disk_t_per)
        plot_vr_ff_overlay(t_per, vr_avg_ff, x1v, idx_inner, disk_ir, disk_t_per)
        plot_vphi_kep_multitime(t_per, vphi_avg_kep, x1v, disk_k, disk_r, k_final)
        plot_vr_ff_multitime(t_per, vr_avg_ff, x1v, disk_k, disk_r, k_final)
   
    print(f"Plots written to: {OUTPUT_DIR}\n")
    _log.close()

if __name__ == "__main__":
    main()
