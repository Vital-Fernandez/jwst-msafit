import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
from msafit.model.config import get_default_config

params = get_default_config()
params['instrument']['filter'] = "CLEAR"
params['instrument']['disperser'] = "PRISM"

disp_curve_stsci = Table.read(f'./data/jwst_nirspec_{params["instrument"]["disperser"].lower()}_disp.fits')
fdisp = interp1d(disp_curve_stsci['WAVELENGTH'],disp_curve_stsci['R'],kind='cubic')
w_prism = np.arange(0.5,6.0,0.05)
R_prism = fdisp(w_prism)

single_fname = 'msafits-Ferruit2022-defaults_PRISM-Rcurve-single_shutter.txt'
wave_arr, R_single = np.loadtxt(single_fname, unpack=True, skiprows=1, delimiter=',')

triple_fname = 'msafits-Ferruit2022-defaults_PRISM-Rcurve-triple_shutter.txt'
wave_arr, R_triple = np.loadtxt(triple_fname, unpack=True, skiprows=1, delimiter=',')

disp_curve_stsci = Table.read(f'./data/jwst_nirspec_{params["instrument"]["disperser"].lower()}_disp.fits')

fig, ax = plt.subplots()
ax.plot(wave_arr, R_prism, label='Original jwst_nirspec_prism_disp.fits')
ax.plot(wave_arr, R_single, label='MSAFIT single shutter')
ax.plot(w_prism, R_triple, label='MSAFIT triple shutter')
ax.set_xlabel(r'Wavelength ($\mu m$)')
ax.set_ylabel('Resolving power')
ax.legend()
plt.savefig('Resolving_power_comparison.png')
plt.show()