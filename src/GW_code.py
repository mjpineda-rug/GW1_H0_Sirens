import h5py
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import corner
from pathlib import Path

# Get path of this script
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR.parent / "data" / "GW170817_GWTC-1.hdf5"
plots_dir = BASE_DIR.parent / "plots"
# Load file
f = h5py.File(data_path, "r")
df = pd.read_hdf(data_path, key='IMRPhenomPv2NRT_highSpin_posterior')

# extract the desired values from the data
d_l = df['luminosity_distance_Mpc']
cos_iota = df['costheta_jn']

z = 0.009877        # redshift from https://ned.ipac.caltech.edu/byname?objname=ngc+4993&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
# z_uncert = 1.67e-5 do we need to take this into account?
c = 299792.458      # speed of light in km/s from https://physics.nist.gov/cgi-bin/cuu/Value?c
planck_mean = 67.4
planck_sigma = 0.5
shoes_mean = 73.0
shoes_sigma = 1.0
np.random.seed(123456)


H_0 = (c * z) / d_l
H_0_grid, cos_iota_grid = np.meshgrid(np.linspace(min(H_0), max(H_0), 100), np.linspace(-1, 1, 100))    # meshgrids used for evaluating the kde on a grid of points
                                                                                                        # cos iota from -1 to 1, H_0 from min to max values of the samples

d_l_grid = (c * z) / H_0_grid   # low redshift approximation: d_L = (c/H_0) * z


""" 
Compute the likelihood of the data given the model parameters (H_0, cos(iota)) using a kernel density estimate (kde) of the data.
Makes use of the low redshift approximation to relate the luminosity distance to H_0 and the redshift.
""" 
bandwidth = 0.1
data_l_d_cos_iota = np.vstack((d_l,cos_iota)) # needed because kde for 2D takes a 2D array of shape (# dims, # data)

# kde model 
kde = gaussian_kde(data_l_d_cos_iota,bw_method=bandwidth)

# apply kde to the grid points
# we need to ravel (flatten) the grid points and stack them into a 2D array of shape (# dims, # grid points) for the kde
kde_values = kde(np.vstack([d_l_grid.ravel(), cos_iota_grid.ravel()]))

# define jacobian for the transformation from (d_L, cos(iota)) to (H_0, cos(iota))
J = (c * z) / (H_0_grid**2)

# need to reshape the kde values back to the shape of the grid and multiply by the jacobian to get the likelihood in terms of (H_0, cos(iota))
likelihood = kde_values.reshape(H_0_grid.shape) * J


def log_likelihood(theta):
    H0, cos_iota_val = theta

    # Extract grid axes
    H0_values = H_0_grid[0, :]
    cos_iota_values = cos_iota_grid[:, 0]

    # Reject out-of-bounds
    if H0 < H0_values.min() or H0 > H0_values.max():
        return -np.inf
    if cos_iota_val < cos_iota_values.min() or cos_iota_val > cos_iota_values.max():
        return -np.inf

    # Find nearest indices
    i = np.abs(cos_iota_values - cos_iota_val).argmin()
    j = np.abs(H0_values - H0).argmin()


    # Extract likelihood value
    like = likelihood[i, j]

    # Handle invalid values
    if like <= 0 or not np.isfinite(like):
        return -np.inf

    return np.log(like)

def log_prior(theta):
    H0, cos_iota = theta
    if 50 < H0 < 100 and -1 <= cos_iota <= 1:
        return -np.log(H0)
    return -np.inf

def sampler_2D(
    n_samples=5000,
    initial=[80, -1],
    proposal_width=[1, 0.05],
):
    accepted_H_0 = []
    accepted_cos_iota = []
    n_accepted = 0
    acceptance_rates = []
    current = np.array(initial)
    chain = [current.copy()]

    for i in range(n_samples):

        # randomly propose new point
        proposal = np.array([
            np.random.normal(current[0], proposal_width[0],),  # H0
            np.random.normal(current[1], proposal_width[1])   # cos(iota)
        ])

        # get log likelihoods for comparison
        log_like_current = log_likelihood(current)
        log_like_proposal = log_likelihood(proposal)

        log_prior_current = log_prior(current)
        log_prior_proposal = log_prior(proposal)

        # set log posterior. because of small values, this is done with the addition of the logs instead of 
        # multiplying likelihoods in order to prevent breakdown
        log_post_current = log_like_current + log_prior_current
        log_post_proposal = log_like_proposal + log_prior_proposal

        # compute acceptance probability, then decide to take current or proposal
        # If proposal is invalid we reject immediately to avoid breakdown
        if not np.isfinite(log_post_current):
            accept = False

        # If current is invalid (should only activate if the very first initials are bad)
        elif not np.isfinite(log_post_proposal):
            accept = False

        else:
            log_p_accept = log_post_proposal - log_post_current
            accept = np.log(np.random.rand()) < log_p_accept

        if accept:
            current = proposal
            n_accepted += 1
    
        accepted_H_0.append(current[0])
        accepted_cos_iota.append(current[1])
        acceptance_rates.append(n_accepted/n_samples)

        chain.append(current.copy())

    return np.array(chain), accepted_H_0, accepted_cos_iota,n_accepted/n_samples

# https://research-portal.uu.nl/ws/portalfiles/portal/248573031/PhysRevD.110.083033.pdf proposal width of 20%
n_samples=100000
initial=[80, -1]
proposal_width = [5, 0.1]

chain = sampler_2D(n_samples,initial,proposal_width)[0]

# Remove burn-in
burnin = int(n_samples/200)
burnin = 100
samples = chain[burnin:]

H0_samples = samples[:, 0]

median = np.median(H0_samples)
low, high = np.percentile(H0_samples, [16, 84])

plt.hist2d(samples[:, 0], samples[:, 1], bins=50, density=True,cmap='YlOrRd')
plt.xlabel(r"$H_0$")
plt.ylabel(r"cos($\iota$)")
plt.colorbar(label="Posterior density")

plt.axvspan(
    shoes_mean - shoes_sigma,
    shoes_mean + shoes_sigma,
    color='blue',
    alpha=0.2,
    label='SH0ES'
)
plt.axvline(shoes_mean, color='blue', linewidth=1)

plt.axvspan(
    planck_mean - planck_sigma,
    planck_mean + planck_sigma,
    color='green',
    alpha=0.2,
    label='Planck'
)
plt.axvline(planck_mean, color='green', linewidth=1)

plt.axvspan(
    low,
    high,
    color='cyan',
    alpha=0.2,
    label=f'68% Credible Region = [{low:.2f},{high:.2f}]'
)
plt.axvline(median, color='black',linestyle='--', linewidth=1,label=f"Median = {median:.2f}")

plt.legend()

plt.savefig(plots_dir / "2D-posterior.png", dpi=300)
plt.close()

fig = corner.corner(
    samples,
    labels=["$H_0$", "cos($\\iota$)"],
    show_titles=True
)
plt.savefig(plots_dir / "2D_and_1D_posterior.png", dpi=300)
plt.close()

print("FILES SAVED IN plots/")