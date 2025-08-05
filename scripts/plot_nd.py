import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import cm
from viscosity_maker import (
    smoothen_viscosity_scipy, write_out_viscosity_profile,
    haskell_measure, new_profile, plot_single)


def __main__():
    my_dir = Path(__file__).parent.parent.resolve()
    # Find all nd_*.txt files in the current directory
    files = [
        # "mu_1_252_254.visc",
        # "mu_2_500_501.visc",
        # "mu_2e20_min_haskell.visc",
        "mu_4e20_min_haskell.visc",
        # "SC2006_400_401.visc",
        # "F10b_lith_305_306.visc",
    ]

    # Giampierro's constraint
    giampierro_constraint = 4e20  # Pa.s

    plt.close('all')
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(111)
    ax.set_position([0.1, 0.1, 0.6, 0.8])
    for i, fname in enumerate([my_dir / f for f in files]):
        # Load data, skipping the first comment line
        _, viscosity = np.loadtxt(
            fname, delimiter=",", skiprows=2, unpack=True)
        radius = np.flipud(np.linspace(
            6370e3 - 2890e3, 6370e3, len(viscosity)))

        # Normalize viscosity to Giampierro's constraint
        giampierro_measure = giampierro_constraint / np.min(viscosity)
        viscosity = viscosity * giampierro_measure
        print(f"{fname.name}: {haskell_measure(radius, viscosity)}")

        # Normalize viscosity
        ax.plot(viscosity, radius, label=fname.name, color=cm.tab10(i))
        # Smooth viscosity using weighted average of neighbors (inverse distance weighting)
        visc_smooth = smoothen_viscosity_scipy(radius, viscosity)
        write_out_viscosity_profile(
            radius, viscosity, fname.with_suffix('.nonsmooth.visc'), nondim=True)
        write_out_viscosity_profile(
            radius, visc_smooth, fname.with_suffix('.smooth.visc'), nondim=True)
        # visc_smooth = smoothen_viscosity_spline(radius, viscosity, smoothing_factor=0.1)
        ax.plot(visc_smooth, radius, linestyle='--',
                label=f"{fname.name} (smoothed) Haskell: {haskell_measure(radius, visc_smooth):.2e}", color=cm.tab10(i))

    ax.set_xscale('log')
    ax.set_xlabel('Viscosity')
    ax.set_ylabel('Radius')
    ax.set_title('Viscosity Profiles')
    ax.legend(loc=(1.01, 0.1))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.savefig(my_dir / "viscosity_profiles.png",
                dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    # __main__()
    plt.close('all')
    fig = plt.figure(figsize=(18, 8), num=1)
    new_profile(Path(__file__).parent.parent.resolve() /
                "./profs_for_rhodri/copy_mu_4e20_min_haskell.nonsmooth.visc")
    plot_single(Path(__file__).parent.parent.resolve() /
                "./profs_for_rhodri/mu_4e20_min_haskell.nonsmooth.visc")
    # plot_single(Path(__file__).parent.resolve() /
    #             "./profs_for_rhodri/copy_mu_4e20_min_haskell.nonsmooth.visc")
    plot_single(Path(__file__).parent.parent.resolve() /
                "./profs_for_rhodri/copy_mu_4e20_min_haskell.nonsmooth.smooth.visc")
