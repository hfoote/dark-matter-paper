#!/usr/bin/env python3
"""
Simulation debris fractions using NEW r_half (from mass-size relation)
and NEW fraction definition: f = M_star(5-20 rh) / (m_star_bound / 2).
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import os

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5-dwarf'

# ---- Load NEW simulation data ----
sats = np.load(os.path.join(OUTDIR, 'sat_arrs_ting_new.npy'))

# Mass cut: M_star_bound > 1e5
mass_cut = sats[:, 2] > 1e5
sats = sats[mass_cut]
print(f"Mass cut M_bound > 1e5: {mass_cut.sum()}/{len(mass_cut)} systems")

f_star_bound = sats[:, 0]
m_star_bound = sats[:, 2]
r_half_kpc = sats[:, 3]    # NEW: from mass-size relation
r_sub_kpc = sats[:, 13]

m_star_lt5 = sats[:, 10]   # SM_bins[2]: <5 rh (new rh)
m_star_lt20 = sats[:, 12]  # SM_bins[4]: <20 rh (new rh)

# NEW fraction: M_star(5-20 rh) / (m_star_bound / 2)
f_5_20 = (m_star_lt20 - m_star_lt5) / (m_star_bound / 2.0)

log_m_star = np.log10(m_star_bound)
m_star_infall = m_star_bound / f_star_bound
log_m_star_infall = np.log10(m_star_infall)
M_V_sim = 4.83 - 2.5 * np.log10(m_star_bound / 1.6)

# ---- Load MW and M31 data ----
mw = Table.read(os.path.join(OUTDIR, 'mw_satellites.csv'))
m31 = Table.read(os.path.join(OUTDIR, 'm31_satellites.csv'))

# ---- Figure: 2x3 panel ----
fig, axes = plt.subplots(2, 3, figsize=(17, 10))

cmap = plt.cm.coolwarm_r
scatter_kw = dict(s=12, alpha=0.6, c=f_star_bound, cmap=cmap, vmin=0, vmax=1, rasterized=True)

f_ref = [0.001, 0.01, 0.1, 1.0]
f_colors = ['C2', 'C3', 'C4', 'C5']

def add_flines(ax):
    for fv, fc in zip(f_ref, f_colors):
        ax.axhline(fv, color=fc, ls='--', alpha=0.5, lw=0.8)

def add_mw_m31_vlines(ax, col):
    for row in mw:
        ax.axvline(row[col], color='C0', alpha=0.08, lw=0.5)
    for row in m31:
        ax.axvline(row[col], color='C1', alpha=0.08, lw=0.5)

# Panel 1: f_5_20 vs f_star_bound
ax = axes[0, 0]
sc = ax.scatter(f_star_bound, f_5_20, **scatter_kw)
ax.set_xlabel(r'$f_{\star,\mathrm{bound}}$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.set_xlim(0, 1.05)
add_flines(ax)

# Panel 2: f_5_20 vs log(M_star_bound)
ax = axes[0, 1]
ax.scatter(log_m_star, f_5_20, **scatter_kw)
add_mw_m31_vlines(ax, 'mass_stellar')
ax.scatter([], [], s=40, marker='o', edgecolors='C0', facecolors='none', label='MW')
ax.scatter([], [], s=40, marker='s', edgecolors='C1', facecolors='none', label='M31')
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{bound}} / M_\odot)$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.legend(fontsize=8, loc='upper left')
add_flines(ax)

# Panel 3: f_5_20 vs log(M_star_infall)
ax = axes[0, 2]
ax.scatter(log_m_star_infall, f_5_20, **scatter_kw)
ax.set_xlabel(r'$\log_{10}(M_{\star,\mathrm{infall}} / M_\odot)$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)

# Panel 4: f_5_20 vs M_V
ax = axes[1, 0]
ax.scatter(M_V_sim, f_5_20, **scatter_kw)
add_mw_m31_vlines(ax, 'M_V')
ax.set_xlabel(r'$M_V$', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
ax.invert_xaxis()
ax.legend(fontsize=8, loc='upper left')
add_flines(ax)

# Panel 5: f_5_20 vs r_half
ax = axes[1, 1]
ax.scatter(r_half_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{1/2}$ [kpc] (from mass-size relation)', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-5, 10)
ax.set_xlim(0.01, 5.0)
add_flines(ax)

# Panel 6: f_5_20 vs distance from host
ax = axes[1, 2]
ax.scatter(r_sub_kpc, f_5_20, **scatter_kw)
ax.set_xlabel(r'$r_{\rm sub}$ [kpc]', fontsize=12)
ax.set_ylabel(r'$f_{5-20}$', fontsize=11)
ax.set_yscale('log')
ax.set_ylim(1e-5, 10)
add_flines(ax)

# Colorbar
fig.subplots_adjust(right=0.88)
cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label(r'$f_{\star,\mathrm{bound}}$', fontsize=12)

fig.suptitle(r'NEW: $f_{5-20} = M_\star(5$-$20\,r_h^{\rm obs}) / (M_{\star,\mathrm{bound}}/2)$'
             r'  |  $M_{\star,\mathrm{bound}} > 10^5\,M_\odot$'
             f'  (N={len(sats)})',
             fontsize=13, y=0.98)
fig.savefig(os.path.join(OUTDIR, 'sim_debris_fraction_new.png'), dpi=150, bbox_inches='tight')
print("Saved sim_debris_fraction_new.png")

# ---- Print statistics ----
print("\n" + "=" * 60)
print("NEW fraction statistics by f_star_bound bins")
print("=" * 60)
bins = [(0.95, 1.01, 'Intact (f_bound > 0.95)'),
        (0.8, 0.95, 'Mildly stripped (0.8 < f_bound < 0.95)'),
        (0.5, 0.8, 'Moderately stripped (0.5 < f_bound < 0.8)'),
        (0.0, 0.5, 'Heavily stripped (f_bound < 0.5)')]

for lo, hi, label in bins:
    mask = (f_star_bound >= lo) & (f_star_bound < hi)
    n = np.sum(mask)
    if n > 0:
        f_vals = f_5_20[mask]
        print(f"\n{label}: N={n}")
        print(f"  f_5_20: median={np.median(f_vals):.4f}, "
              f"mean={np.mean(f_vals):.4f}, "
              f"16-84%=[{np.percentile(f_vals, 16):.5f}, {np.percentile(f_vals, 84):.4f}]")

print("\n" + "=" * 60)
print("NEW fraction statistics by log(M_star) bins")
print("=" * 60)
mbins = [(2, 4, 'UFDs (log M* = 2-4)'),
         (4, 6, 'Classical dwarfs (log M* = 4-6)'),
         (6, 8, 'Bright dwarfs (log M* = 6-8)')]

for lo, hi, label in mbins:
    mask = (log_m_star >= lo) & (log_m_star < hi)
    n = np.sum(mask)
    if n > 0:
        f_vals = f_5_20[mask]
        print(f"\n{label}: N={n}")
        print(f"  f_5_20: median={np.median(f_vals):.4f}, "
              f"mean={np.mean(f_vals):.4f}, "
              f"16-84%=[{np.percentile(f_vals, 16):.5f}, {np.percentile(f_vals, 84):.4f}]")
