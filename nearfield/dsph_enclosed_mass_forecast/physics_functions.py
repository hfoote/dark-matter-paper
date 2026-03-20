"""
physics_functions.py
====================
Standalone documented physics functions used in the SIDM half-light
radius mass function analysis. All functions are self-contained with
explicit units in docstrings.

References
----------
Yang+2023   : Yang, Nadler, Yu, Zhong 2024, JCAP 2, 032 (arXiv:2305.16176)
Nadler+2023 : Nadler, Yang, Yu 2023, ApJL 958, L20 (arXiv:2306.01830)
Wolf+2010   : Wolf et al. 2010, MNRAS 406, 1220
Gehrels1986 : Gehrels 1986, ApJ 303, 336
Tinker+2008 : Tinker et al. 2008, ApJ 688, 709
Diemer+2019 : Diemer & Joyce 2019, ApJ 871, 168
"""

import numpy as np
from scipy.integrate import dblquad
from scipy.stats import chi2 as chi2_dist

# Gravitational constant in units of kpc^3 M_sun^-1 Gyr^-2
GG = 4.30073e-6


# ── Cross section ─────────────────────────────────────────────────────────

def sigmaVis(v, sigma0, w):
    """
    Viscosity (momentum-transfer) cross section for Rutherford-like scattering.

    Parameters
    ----------
    v      : float  Relative velocity [km/s]
    sigma0 : float  Cross section normalisation [cm²/g]
    w      : float  Velocity turnover scale [km/s]

    Returns
    -------
    sigma_vis : float  [cm²/g]

    Notes
    -----
    Differential cross section: dσ/dΩ = σ₀ w⁴ / [2(w² + v² sin²(θ/2))²]
    The viscosity cross section weights by sin²θ and integrates over angles.
    See Yang & Yu 2022, Yang+2023 Eq.(1.1).
    """
    return (
        3.0 * sigma0 * w**6
        * (-4*v**2/w**2 + 2*(2 + v**2/w**2)*np.log(1 + v**2/w**2))
        / v**6
    )


def sigmaeff(vmax, sigma0, w):
    """
    Thermally-averaged effective cross section for a Rutherford-like model.

    Uses direct double integration over (θ, v) with a Maxwell-Boltzmann
    velocity distribution characterised by ν_eff = 1.1 × vmax / √3.

    Parameters
    ----------
    vmax   : float  V_max of the NFW halo [km/s]
    sigma0 : float  Cross section normalisation [cm²/g]
    w      : float  Velocity turnover scale [km/s]

    Returns
    -------
    sigma_eff : float  [cm²/g]

    Notes
    -----
    v_upper = 10 × max(vmax, w) ensures the cross section has turned over
    before the integration is truncated. Yang+2023 use v_upper = 3.06 × vmax
    which underestimates σ_eff when vmax << w (dwarf regime).
    See Yang+2023 Eq.(1.1); Nadler+2023 Eq.(2).
    """
    veff    = 1.1 * vmax / np.sqrt(3)
    v_upper = 10.0 * max(vmax, w)
    diff    = lambda theta, v: sigma0 * w**4 / (2*(w**2 + v**2*np.sin(theta/2)**2)**2)
    fup     = lambda v, theta: v**7 * np.sin(theta)**3 * np.exp(-v**2/(4*veff**2)) * diff(theta, v)
    fdown   = lambda v, theta: v**7 * np.sin(theta)**3 * np.exp(-v**2/(4*veff**2))
    termu   = dblquad(fup,   0, np.pi, 0, v_upper, epsrel=1e-4)[0]
    termd   = dblquad(fdown, 0, np.pi, 0, v_upper, epsrel=1e-4)[0]
    return 2 * termu / termd


# ── Core-collapse timescale ───────────────────────────────────────────────

def tc_dmo(sigma_eff_val, rhos, rs):
    """
    Gravothermal core-collapse timescale (dark matter only).

    Parameters
    ----------
    sigma_eff_val : float  Effective cross section [cm²/g]
    rhos          : float  NFW characteristic density [M_sun/kpc³]
    rs            : float  NFW scale radius [kpc]

    Returns
    -------
    tc : float  Core-collapse time [Gyr]

    Notes
    -----
    t_c = (150/C) / (σ_eff × 2.09×10⁻¹⁰ × ρ_s × r_s) / √(4πG ρ_s)
    with C = 0.75 calibrated against N-body simulations.
    The factor 2.09×10⁻¹⁰ converts units:
      [cm²/g × M_sun/kpc³ × kpc]^{-1} × [kpc³/(M_sun Gyr²)]^{-1/2} → Gyr
    See Yang+2023 Eq.(2.2); parametricC4.py tc() with a=0.
    """
    return (
        (150.0 / 0.75)
        / (sigma_eff_val * 2.09e-10 * rhos * rs)
        / np.sqrt(4.0 * np.pi * GG * rhos)
    )


# ── Formation redshift ─────────────────────────────────────────────────────

def formation_redshift(M_msun):
    """
    Median halo formation redshift as a function of virial mass.

    Parameters
    ----------
    M_msun : float  Halo virial mass [M_sun]

    Returns
    -------
    z_form : float  Median formation redshift

    Notes
    -----
    Quadratic fit to the Giocoli+2011 relation, calibrated to ΛCDM simulations.
    Formation redshift defined as when the main progenitor first assembled
    half the final halo mass.
    z_form = -0.0064 x² - 0.1043 x + 1.4807  where x = log10(M / 10^10 M_sun)
    """
    x = np.log10(M_msun / 1e10)
    return -0.0064*x**2 - 0.1043*x + 1.4807


def tlb(z):
    """
    Lookback time to redshift z.

    Parameters
    ----------
    z : float  Redshift

    Returns
    -------
    t : float  Lookback time [Gyr]

    Notes
    -----
    Analytic approximation calibrated for Planck-like cosmology
    (Ωm=0.286, ΩΛ=0.714, h=0.7). Copied from parametricC4.py.
    Accurate to <0.1% for 0 < z < 10.
    """
    return (13.647247606199668
            - 11.020482589612016
            * np.log(1.5800327517186143/(1+z)**1.5
                     + np.sqrt(1.0 + 2.4965034965034962/(1+z)**3)))


def halo_age(M_msun):
    """
    Median age of a halo of mass M_msun [M_sun] at z=0.

    Returns lookback time to formation redshift [Gyr].
    """
    return tlb(formation_redshift(M_msun))


# ── Half-light radius ─────────────────────────────────────────────────────

def r_half_pc(Rvir_kpc, scatter_dex=0.63, A=37.0, N=1.07,
              rng=None, clip_min=20.0, clip_max=3000.0):
    """
    Stellar half-light radius in parsecs from virial radius.

    Parameters
    ----------
    Rvir_kpc   : float or array  Virial radius [kpc]
    scatter_dex: float           Log-normal scatter [dex]
    A          : float           Normalisation [pc at Rvir=10 kpc]
    N          : float           Power-law slope
    rng        : np.random.Generator or None
    clip_min   : float           Minimum r_half [pc]
    clip_max   : float           Maximum r_half [pc]

    Returns
    -------
    r_half : float or array  Half-light radius [pc]

    Notes
    -----
    r_{1/2} = A × (Rvir/10)^N × 10^{N(0, scatter_dex)}
    Fitted to observed dSph size-halo size relation.
    A=37 pc at Rvir=10 kpc, N=1.07, scatter=0.63 dex.
    Physical bounds clip to [20, 3000] pc matching observed dSph range.
    """
    if rng is None:
        rng = np.random.default_rng()
    scatter = 10.0**rng.normal(0, scatter_dex, np.shape(Rvir_kpc))
    r = A * (np.asarray(Rvir_kpc) / 10.0)**N * scatter
    return np.clip(r, clip_min, clip_max)


# ── Wolf mass estimator ───────────────────────────────────────────────────

def wolf_mass(sigma_los_kms, r_half_pc_val):
    """
    Wolf+2010 mass estimator: M(<r_{1/2}) = 5 σ²_los r_{1/2} / G

    Parameters
    ----------
    sigma_los_kms : float  Line-of-sight velocity dispersion [km/s]
    r_half_pc_val : float  Projected half-light radius [pc]

    Returns
    -------
    M_half : float  Enclosed mass [M_sun]

    Notes
    -----
    G = 4.302×10⁻³ pc M_sun^{-1} (km/s)²
    Wolf et al. 2010, MNRAS 406, 1220, Eq.(2).
    Valid for dispersion-supported systems with arbitrary anisotropy.
    """
    G_pc = 4.302e-3   # pc (km/s)² M_sun^{-1}
    return 5.0 * sigma_los_kms**2 * r_half_pc_val / G_pc


def wolf_mass_frac_error(sigma_los, sigma_err, r_half_pc_val, r_half_err_pc=0.0):
    """
    Fractional uncertainty on the Wolf mass estimate.

    Parameters
    ----------
    sigma_los     : float  Line-of-sight velocity dispersion [km/s]
    sigma_err     : float  Uncertainty on sigma_los [km/s]
    r_half_pc_val : float  Half-light radius [pc]
    r_half_err_pc : float  Uncertainty on r_half [pc] (often small)

    Returns
    -------
    frac_err : float  Fractional mass uncertainty (ΔM/M)

    Notes
    -----
    Error propagation: (ΔM/M)² = (2Δσ/σ)² + (Δr/r)²
    r_half errors are typically subdominant.
    """
    return np.sqrt((2*sigma_err/sigma_los)**2 + (r_half_err_pc/r_half_pc_val)**2)


# ── Poisson statistics ────────────────────────────────────────────────────

def gehrels_errors(counts, alpha=0.317):
    """
    Gehrels (1986) Poisson confidence intervals for small counts.

    Parameters
    ----------
    counts : array-like  Observed counts per bin
    alpha  : float       Significance level (0.317 for 1σ)

    Returns
    -------
    lo, hi : arrays  Lower and upper error bars

    Notes
    -----
    Uses chi² distribution for exact Poisson intervals.
    Handles N=0 bins correctly (lo=0, hi from upper interval).
    Gehrels 1986, ApJ 303, 336.
    """
    counts = np.asarray(counts, dtype=float)
    hi = 0.5*chi2_dist.ppf(1-alpha/2, 2*(counts+1)) - counts
    lo = counts - 0.5*chi2_dist.ppf(alpha/2, 2*np.maximum(counts, 1))
    return np.where(counts==0, 0, lo), hi


# ── NFW profile ───────────────────────────────────────────────────────────

def nfw_vmax(rhos, rs):
    """
    Maximum circular velocity of an NFW halo.

    Parameters
    ----------
    rhos : float  Characteristic density [M_sun/kpc³]
    rs   : float  Scale radius [kpc]

    Returns
    -------
    vmax : float  [km/s]

    Notes
    -----
    V_max = 1.648 × r_s × √(G ρ_s)
    For an NFW profile, V_max occurs at R_max = 2.1626 × r_s.
    """
    return 1.648 * rs * np.sqrt(GG * rhos)


def nfw_enclosed_mass(r, rhos, rs):
    """
    Analytic NFW enclosed mass within radius r.

    Parameters
    ----------
    r    : float  Radius [kpc]
    rhos : float  Characteristic density [M_sun/kpc³]
    rs   : float  Scale radius [kpc]

    Returns
    -------
    M_enc : float  [M_sun]
    """
    x = r / rs
    return 4.0 * np.pi * rhos * rs**3 * (np.log(1.0 + x) - x/(1.0 + x))
