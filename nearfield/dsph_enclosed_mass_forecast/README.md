# SIDM Half-Light Radius Mass Function

## Overview

This repository implements a Monte Carlo forward model for the enclosed dark matter
mass function within stellar half-light radii of dwarf spheroidal galaxies, comparing
cold dark matter (CDM), self-interacting dark matter (SIDM), warm dark matter (WDM),
and power-spectrum-bump models against the observed Local Group dSph population.

The key observable is M(<r_{1/2}): the dark matter mass enclosed within the stellar
half-light radius, estimated via the Wolf+2010 mass estimator from line-of-sight
velocity dispersions. This quantity is relatively insensitive to the assumed dark
matter profile and is directly measurable from spectroscopy.

## Repository Structure

```
sidm_halflight_final_clean.ipynb   # Main analysis notebook (17 cells)
tests_sidm.ipynb                   # Unit tests for all pipeline components
physics_functions.py               # Standalone documented physics functions
README.md                          # This file

parametricSIDM/                    # Clone from github.com/DanengYang/parametricSIDM
dsph_wolf_mass_forecasts.csv       # SpecS5 Wolf mass forecasts for LG dSphs
dsph_wolf_mass_lit.csv             # Literature Wolf masses and uncertainties
cdm_data.txt                       # External CDM N(>M) mass function
wdm7_data.txt                      # WDM 7 keV N(>M) mass function
bump1_data.txt                     # Bump model 1 N(>M) mass function
bump2_data.txt                     # Bump model 2 N(>M) mass function (bump3_data.txt in code)
```

## Dependencies

```bash
conda create -n sidm python=3.10 -y
conda activate sidm
conda install numpy scipy matplotlib pandas jupyter ipykernel tqdm -y
pip install colossus
pip install "scipy<1.14"   # parametricSIDM uses scipy.misc.derivative (removed in 1.14)
                            # alternatively patch parametricCoredDZ.py line 10

git clone https://github.com/DanengYang/parametricSIDM.git
python -m ipykernel install --user --name sidm --display-name "Python (sidm)"
```

## Key Physical Choices and References

| Component | Choice | Reference |
|-----------|--------|-----------|
| Cosmology | Planck18 | Planck Collaboration 2018 |
| HMF | Tinker+2008, mdef=200c | Tinker et al. 2008, ApJ 688, 709 |
| c(M) relation | Diemer-Joyce 2019, 0.15 dex scatter | Diemer & Joyce 2019, ApJ 871, 168 |
| NFW params | colossus NFWProfile | Diemer 2018 |
| Rvir definition | Bryan-Norman 1998 overdensity | Bryan & Norman 1998, ApJ 495, 80 |
| r_{1/2} relation | 37×(Rvir/10)^1.07 pc, 0.63 dex scatter | Jiang et al. 2019 |
| r_{1/2} bounds | 20–3000 pc | Observed dSph range |
| z_form | Quadratic fit to Giocoli+2011 | Giocoli et al. 2011 |
| σ_eff integral | Direct double integral over (θ,v), v_upper=10×max(vmax,w) | Yang & Yu 2022 |
| tc formula | (150/0.75)/(σ_eff×2.09e-10×ρ_s×r_s)/√(4πGρ_s) | Yang+2023, Eq.(2.2) |
| Profile evolution | rhost/rst/rct from parametricCoredDZ | Yang+2023, Eq.(2.3) |
| M_enc approximation | tanh(r/rc) × M_NFW(evolved) | Adapted from Yang+2023 |
| Wolf mass estimator | M(<r_{1/2}) = 5σ²r_{1/2}/G | Wolf et al. 2010 |
| Poisson errors | Gehrels 1986 intervals | Gehrels 1986, ApJ 303, 336 |
| Rubin forecast | 170 (150–200) within 800 kpc | Tsiane+2025; Mau et al. 2021 |

## SIDM Models Implemented

The SIDM cross section follows a Rutherford-like velocity dependence:
  dσ/dΩ = σ₀ w⁴ / [2(w² + v² sin²(θ/2))²]

Two benchmark models are included:
- **σ₀=147, w=70 km/s**: Moderate interaction, produces cores in lower-mass halos
- **σ₀=500, w=50 km/s**: Strong interaction, maximum diversity of core/collapse phases

## Running the Analysis

1. Ensure all data files are in the same directory as the notebook
2. Set `PSIDM_ROOT` in Cell 1 to point to your parametricSIDM clone
3. Run cells sequentially (1–15)
4. Cell 8 (SIDM runs) takes ~10–20 minutes due to the σ_eff double integrals
5. Output PDFs: plot1_cumulative_mf.pdf, plot2_mass_ratio.pdf, plot3_rubin_forecast.pdf

## Known Limitations

- Uses the "basic approach" of parametricSIDM (constant halo age per mass bin),
  not the integral approach which tracks full merger history
- r_{1/2} relation has large scatter (0.63 dex) that dominates over SIDM signal
  in the binned histogram; individual galaxy measurements are more sensitive
- The model treats field halos from a cosmological HMF; observed LG dSphs are
  satellites with tidal histories not modelled here
- Normalisation of model curves to observed total is shape-only; absolute amplitude
  requires a galaxy-halo connection model
- Completeness correction not applied (volume-complete but not luminosity-complete)
