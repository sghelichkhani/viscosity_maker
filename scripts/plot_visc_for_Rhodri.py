from pathlib import Path
import numpy as np

data_dir = Path("/Users/sghelichkhani/Data/ADJOINT/VISC_PROFS/profiles")

visc_files = ["F10b_lith_305_306.visc", "mu_1_252_254.visc",
              "mu_2_500_501.visc", "SC2006_400_401.visc"]
all_profs = {}
for fi in visc_files:
    rad, visc = np.loadtxt(data_dir / fi, delimiter=",", skiprows=2,
                           usecols=(0, 1), unpack=True)
    non_d_rad = np.flipud(np.linspace(1.208, 2.208, len(rad)))
    output_str = "\n".join(
        [f"{r_nd:.4f}, {v/visc.min():.3e}" for r_nd, v in zip(non_d_rad, visc)])
    with open("nd_" + fi, mode="w") as new_fi:
        new_fi.write(f"# min_visc = {visc.min():.3e}, Viscosity\n")
        new_fi.write(output_str)
