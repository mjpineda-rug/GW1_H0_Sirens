# GW1_H0_Sirens

Authors: Elisa Genot / s5532930, Elliot Craig / s5617863, Leticia Arruda / , Michael Pineda / s5944163, Horia-Emilian Nita / s5756758

# Gravitational-Wave Standard Siren Measurement of $H_0$

## Project Overview

This project estimates the Hubble constant $H_0$ using gravitational-wave data from the binary neutron star merger GW170817. Using the standard siren method, we combine the luminosity distance posterior from LIGO/Virgo with the redshift of the host galaxy NGC 4993. Bayesian inference is performed using MCMC to obtain the posterior distribution of $H_0$ and the inclination angle.

---

## Setup

Install the required Python packages:

```bash
pip install numpy scipy matplotlib h5py corner
```

---

## Repository Structure

```
project/
│
├── data/          # GW170817 posterior samples
├── notebooks/     # Jupyter notebooks for analysis
├── src/           # Core code
├── plots/         # Generated figures
└── README.md
```

---

## Data

Download the GW170817 posterior samples from:

https://dcc.ligo.org/LIGO-P1800370/public

Place the downloaded file in the `data/` folder.

---

## How to Run

Both the python scripts and the analysis notebook will 

- Load the GW posterior samples
- Construct the likelihood using a KDE
- Run MCMC to sample the posterior
- Generate two dimensional distribution of $H_0$ and $\cos(\iota)$

The analysis notebook additionally produces a number of intermediate plots, useful for selecting parameters and observing algorithm performance.

To run the analysis notebook:

```bash
jupyter notebook notebooks/GW_notebook.ipynb
```

To run the python script:

```bash
python src/GW_code.py
```

---

## Outputs

The python script produces:

- 2D posterior distribution of $H_0$ and $\cos(\iota)$
- Separate 1D posterior distributions of $H_0$ and $\cos(\iota)$
- Comparison with Planck and SH0ES measurements

The analysis notebook additionally produces:

- Distribution of the original LIGO/Virgo data
- Likelihood surfaces for various KDE bandwidths
- MCMC traces for various proposal widths

All plots are saved in the `plots/` folder.

---

## Method Summary

* The luminosity distance $d_L$ is related to $H_0$ via the Hubble–Lemaître law.
* A Kernel Density Estimate (KDE) and the flat-in-log prior $p(H_0) \propto 1/H_0$ are used to model the posterior distribution of $H_0$ and the inclination angle $\iota$.
* A Jacobian factor is included to transform from distance to (H_0).
* MCMC sampling is used to compute the posterior distribution.

---

## Assumptions

* Low-redshift approximation is used.
* Peculiar velocity uncertainty is modeled as a Gaussian with (\sigma \approx 250) km/s.
* A flat-in-log prior $p(H_0) \propto 1/H_0$ is used.

---

## Results

The two dimensional posterior distribution produced by the MCMC chain has a large 68% credible region, which is to be expected given the extremely small sample size. This region is consistent with the values from both Planck 2018 and SH0ES.