import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from viscosity_maker import read_viscosity_profile, smoothen_viscosity_scipy, haskell_measure, smoothen_gaussian
from viscosity_maker.utilities import write_out_viscosity_profile


def make_linear_relationship(x_0, y_0, x_1, y_1):
    a = (y_1 - y_0) / (x_1 - x_0)
    b = y_0 - a * x_0
    return lambda x: a * x + b


def manual_viscosity_profile():

    jumps = {
        f"{3000:0.0f}": make_linear_relationship(2590e3, np.log10(1e23), 2890e3, np.log10(5e22)),
        f"{2890 - 300:0.0f}": make_linear_relationship(700e3, np.log10(1e22), 2590e3, np.log10(1e23)),
        f"{700:0.0f}": lambda x: np.log10(2e21),
        f"{360:0.0f}": lambda x: np.log10(2e20),
        f"{100:0.0f}": lambda x: np.log10(1e23),
    }
    dpths = np.linspace(0, 2890.e3, 257)
    rads = 6370.e3 - dpths
    visc = np.ones(len(dpths))

    for key, value in jumps.items():
        visc[dpths < float(key) * 1e3] = 10**value(dpths[dpths < float(key) * 1e3])

    return rads, visc


filename = Path(__file__).parent.parent.resolve() / \
    "profiles/mu_4e20_min_cmb_1e22_haskell.smooth.visc"
rad, visc, header = read_viscosity_profile(filename)

man_rads, man_visc = manual_viscosity_profile()
new_haskell_measure = haskell_measure(man_rads, man_visc)
man_rads /= man_rads.max() / 2.208

man_visc = smoothen_gaussian(man_rads, man_visc, sigma=0.01)


write_out_viscosity_profile(man_rads, man_visc, "mu_2e20_asthenosphere_linear_increase.visc", nondim=True)
fig = plt.figure(figsize=(10, 5), num=1)
fig.clear()

ax = fig.add_subplot(111)
ax.plot(visc, rad, label="mu_4e20_min_cmb_1e22_haskell")
ax.plot(man_visc, man_rads, label="New one")
ax.set_xscale("log")
ax.text(x=0.5, y=0.95, s=f"Haskell: {new_haskell_measure:.2e}",
        transform=ax.transAxes, ha="center", va="top")
ax.grid()
ax.legend()
fig.show()
