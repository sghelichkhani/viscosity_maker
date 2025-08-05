import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from pathlib import Path


def smoothen_viscosity_spline(radius, viscosity, smoothing_factor=0.1):
    """
    Smooth the viscosity profile using cubic spline interpolation.
    """
    # Create a cubic spline interpolation of the viscosity data
    cs = UnivariateSpline(np.flipud(radius) / 1e6,
                          np.flipud(np.log10(viscosity)), s=smoothing_factor)
    # Evaluate the spline at the original radius points
    visc_smooth = cs(radius / 1e6)
    return 10 ** visc_smooth


# Create a KDTree for efficient nearest neighbor search
def smoothen_gaussian(radius, viscosity, sigma=0.05):
    """
    Smooth the viscosity profile using a Gaussian convolution.

    Parameters:
    -----------
    radius : array_like
        The radius values in meters.
    viscosity : array_like
        The viscosity values to smooth.
    sigma : float, optional
        Standard deviation of the Gaussian kernel in normalized radius units.
        Default is 0.05 (about 50 km).

    Returns:
    --------
    array_like
        Smoothed viscosity profile.
    """
    # Normalize radius to [0, 1] for better kernel scaling
    r_norm = (radius - radius.min()) / (radius.max() - radius.min())

    # Work in log space for viscosity
    log_visc = np.log10(viscosity)

    # Initialize smoothed array
    log_visc_smooth = np.zeros_like(log_visc)

    # Apply Gaussian convolution
    for i in range(len(radius)):
        # Calculate Gaussian weights
        weights = np.exp(-0.5 * ((r_norm - r_norm[i]) / sigma)**2)

        # Normalize weights
        weights = weights / np.sum(weights)

        # Apply weighted average
        log_visc_smooth[i] = np.sum(weights * log_visc)

    # Convert back from log space
    return 10**log_visc_smooth


def smoothen_viscosity_scipy(radius, viscosity):
    """
    Smooth the viscosity profile using inverse distance weighting.
    """
    # Create a KDTree for efficient nearest neighbor search
    tree = cKDTree(radius[:, np.newaxis] / 1e6)
    dists, inds = tree.query(radius[:, np.newaxis] / 1e6, k=11)
    # dists[:, 0] += 0.1  # Avoid division by zero
    weights = 1 / np.exp(dists)   # Inverse distance weighting
    visc_smooth = np.sum(np.log10(viscosity[inds]) * weights,
                         axis=1) / np.sum(weights, axis=1)
    return 10 ** visc_smooth


def haskell_measure(radius, viscosity):
    """
    Calculate the Haskell measure for the viscosity profile.
    """
    # Calculate the Haskell measure as the ratio of the viscosity at 1000 km
    # to the viscosity at 150 km from the surface
    radius_lith = 6370e3 - 100e3  # 150 km from the surface
    radius_UM_LM = 6370e3 - 600e3  # 1000 km from the surface
    logical_array = np.logical_and(radius < radius_lith, radius > radius_UM_LM)
    return np.average(viscosity[logical_array])


def read_viscosity_profile(filename: Path):
    """
    Write the viscosity profile to a file.
    """
    header = {}

    with open(filename, "r") as f:
        first_line = f.readline()
        if first_line.startswith("#"):
            for term in first_line[1:].split(","):
                header[term.split("=")[0].strip()] = float(term.split("=")[-1])

    radius, viscosity = np.loadtxt(filename, delimiter=",", skiprows=1, unpack=True)
    print(header.get("mu_r"))
    return radius, viscosity * header.get("mu_r", 1.0), header


def write_out_viscosity_profile(radius, viscosity, filename, nondim=True):
    """
    Write the viscosity profile to a file.
    """
    to_write_radius = np.flipud(
        np.linspace(1.208, 2.208, len(viscosity)) if nondim else radius
    )

    out_str = "\n".join(
        [
            f"{r:.3f}, {mu:.2e}"
            for r, mu in zip(to_write_radius, viscosity/np.min(viscosity) if nondim else 1)
        ]
    )

    nd_dic = non_dims()
    nd_dic["mu_r"] = viscosity.min()  # minimum value of viscosity

    with open(filename, mode="w") as f:
        f.write(f"# mu_r = {nd_dic['mu_r']:.2e}, Ra = {nd_dic['Ra']:.2e}\n")
        f.write(out_str)


def non_dims():
    ret = {
        "b": 2890e3,  # in meters
        "Delta_T_r": 4000 - 300 - 935,  # temperature difference in Kelvin
        "rho_r": 3200,  # density in kg/m^3
        "g_r": 9.81,  # gravitational acceleration in m/s^2
        "alpha_r": 4.177e-5,  # thermal expansion coefficient in 1/K
        "c_p_r": 1.2497e3,  # specific heat at constant pressure in J/(kg·K)
        "mu_r": 4e20,  # viscosity in Pa·s
        "k_r": 4.0
    }
    # Rayleigh number calculation
    ret["Ra"] = compute_Ra(ret)
    # diffusivity
    ret["kappa_r"] = ret["k_r"] / (ret["rho_r"] * ret["c_p_r"])
    return ret


def compute_Ra(ret):
    return (
        (ret["alpha_r"] * ret["Delta_T_r"] *
         ret["rho_r"]**2 * ret["g_r"] *
         ret["b"]**3 * ret["c_p_r"]) / (ret["mu_r"] * ret["k_r"])
    )


def plot_single(fname):
    """
    Plot a single viscosity profile from a file.
    """
    radius, viscosity = np.loadtxt(
        fname, delimiter=",", skiprows=1, unpack=True)
    plt.plot(4e20 * viscosity, radius, label=fname.name)
    plt.xscale('log')
    plt.xlabel('Viscosity')
    plt.ylabel('Radius')
    plt.title('Viscosity Profile')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()


def new_profile(fname):
    """
    Plot a single viscosity profile from a file.
    """
    radius, viscosity = np.loadtxt(
        fname, delimiter=",", skiprows=1, unpack=True)
    viscosity_smooth = smoothen_gaussian(radius, viscosity, sigma=0.01)
    write_out_viscosity_profile(radius=radius, viscosity=viscosity_smooth * 4e20,
                                filename=fname.with_suffix('.smooth.visc'))
