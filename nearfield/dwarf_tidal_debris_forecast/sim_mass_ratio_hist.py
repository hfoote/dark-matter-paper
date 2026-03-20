#!/usr/bin/env python3
"""
Histogram of M_star(5-20 rh) / M_star_bound for all systems in the NEW simulation.
This shows what fraction of the total bound stellar mass lies in the 5-20 rh annulus.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTDIR = '/Users/tingli/Dropbox/claude_research/spec-s5-dwarf'

# ---- Load NEW simulation data ----
sats = np.load(os.path.join(OUTDIR, 'sat_arrs_ting_new.npy'))

f_star_bound = sats[:, 0]
m_star_bound = sats[:, 2]
m_star_lt5 = sats[:, 10]   # SM_bins[2]: <5 rh (new rh)
m_star_lt20 = sats[:, 12]  # SM_bins[4]: <20 rh (new rh)

# Mass cut: M_star_bound > 1e5
mass_cut = m_star_bound > 1e5
print(f"Total systems: {len(m_star_bound)}, with M_bound > 1e5: {mass_cut.sum()}")
f_star_bound = f_star_bound[mass_cut]
m_star_bound = m_star_bound[mass_cut]
m_star_lt5 = m_star_lt5[mass_cut]
m_star_lt20 = m_star_lt20[mass_cut]

# Ratio: M_star(5-20 rh) / M_star_bound
m_5_20 = m_star_lt20 - m_star_lt5
ratio = m_5_20 / m_star_bound

print(f"After cut: {len(ratio)} systems")
print(f"M(5-20rh)/M_bound: median={np.median(ratio):.4f}, "
      f"mean={np.mean(ratio):.4f}")
print(f"  16-84%: [{np.percentile(ratio, 16):.4f}, {np.percentile(ratio, 84):.4f}]")
print(f"  min={np.min(ratio):.4f}, max={np.max(ratio):.4f}")

# Also show M(<5rh)/M_bound, M(<20rh)/M_bound, M(>20rh)/M_bound
ratio_lt5 = m_star_lt5 / m_star_bound
ratio_lt20 = m_star_lt20 / m_star_bound
ratio_gt20 = (m_star_bound - m_star_lt20) / m_star_bound
print(f"\nM(<5rh)/M_bound: median={np.median(ratio_lt5):.4f}, "
      f"mean={np.mean(ratio_lt5):.4f}")
print(f"M(<20rh)/M_bound: median={np.median(ratio_lt20):.4f}, "
      f"mean={np.mean(ratio_lt20):.4f}")
print(f"M(>20rh)/M_bound: median={np.median(ratio_gt20):.4f}, "
      f"mean={np.mean(ratio_gt20):.4f}")

# ---- Histogram ----
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Panel 1: M(5-20rh)/M_bound histogram
ax = axes[0]
ax.hist(ratio, bins=50, range=(0, 1.2), color='C0', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(np.median(ratio), color='r', ls='--', lw=1.5,
           label=f'median = {np.median(ratio):.3f}')
ax.set_xlabel(r'$M_\star(5$-$20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(r'$M_\star(5$-$20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=12)
ax.legend(fontsize=10)

# Panel 2: M(<5rh)/M_bound histogram
ax = axes[1]
ax.hist(ratio_lt5, bins=50, range=(0, 1.2), color='C1', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(np.median(ratio_lt5), color='r', ls='--', lw=1.5,
           label=f'median = {np.median(ratio_lt5):.3f}')
ax.set_xlabel(r'$M_\star(<5\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(r'$M_\star(<5\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=12)
ax.legend(fontsize=10)

# Panel 3: M(<20rh)/M_bound histogram
ax = axes[2]
ax.hist(ratio_lt20, bins=50, range=(0, 1.2), color='C2', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(np.median(ratio_lt20), color='r', ls='--', lw=1.5,
           label=f'median = {np.median(ratio_lt20):.3f}')
ax.set_xlabel(r'$M_\star(<20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(r'$M_\star(<20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=12)
ax.legend(fontsize=10)

# Panel 4: M(>20rh)/M_bound histogram
ax = axes[3]
ax.hist(ratio_gt20, bins=50, range=(0, 1.2), color='C3', alpha=0.7, edgecolor='k', lw=0.3)
ax.axvline(np.median(ratio_gt20), color='r', ls='--', lw=1.5,
           label=f'median = {np.median(ratio_gt20):.3f}')
ax.set_xlabel(r'$M_\star(>20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.set_title(r'$M_\star(>20\,r_h) / M_{\star,\mathrm{bound}}$', fontsize=12)
ax.legend(fontsize=10)

fig.suptitle(r'Mass ratios using NEW $r_h$ (from mass-size relation) | $M_{\star,\mathrm{bound}} > 10^5\,M_\odot$', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'sim_mass_ratio_hist.png'), dpi=150, bbox_inches='tight')
print('\nSaved sim_mass_ratio_hist.png')

# ---- Breakdown by stripping ----
print("\n" + "=" * 60)
print("Breakdown by f_star_bound")
print("=" * 60)
bins = [(0.95, 1.01, 'Intact (f_bound > 0.95)'),
        (0.8, 0.95, 'Mildly stripped (0.8-0.95)'),
        (0.5, 0.8, 'Moderately stripped (0.5-0.8)'),
        (0.0, 0.5, 'Heavily stripped (f_bound < 0.5)')]

for lo, hi, label in bins:
    mask = (f_star_bound >= lo) & (f_star_bound < hi)
    n = np.sum(mask)
    if n > 0:
        r = ratio[mask]
        print(f"\n{label}: N={n}")
        print(f"  M(5-20rh)/M_bound: median={np.median(r):.4f}, "
              f"mean={np.mean(r):.4f}, "
              f"16-84%=[{np.percentile(r, 16):.4f}, {np.percentile(r, 84):.4f}]")
