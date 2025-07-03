from pathlib import Path

from msafit.model.config import get_default_config
from msafit.lsf.fit_func import gauss_dataset
from matplotlib.ticker import AutoMinorLocator
from matplotlib import pyplot as plt
from msafit.fpa import PSFLib
from msafit.model import Sersic
from msafit.fpa import Spec2D
import numpy as np
#
# # # let's inspect the parameters in the default config dict
# # params = get_default_config()
# # print(params.keys())
# # print("\nparams['instrument']",params["instrument"])
# # print("\nparams['geometry']=", params["geometry"])
# # print("\nparams['morph']=", params["morph"])
# #
# # params['instrument']['filter'] = "CLEAR"
# # params['instrument']['disperser'] = "PRISM"
# #
# # psfs = PSFLib('1x3_PRISM_Q3_PSFLib.fits')
# # params["grid"]["x_grid"] = psfs.psf_x
# # params["grid"]["y_grid"] = psfs.psf_y
# # params["grid"]["wave_grid"] = psfs.psf_wave
# #
# # # create a model cube I(x,y,lambda)
# # model = Sersic(params)
# # model(params["grid"]["wave_grid"])
# #
# # print(model.data.shape)
# # spec = Spec2D(params)
# #
# x1d,spec1d = spec.to_1d()
# #
# # model_1d = gauss_dataset(fit_output[0].params,0,len(lsf.wl_fpa),x1d)
from msafit.lsf import LSF
from copy import deepcopy

from astropy.table import Table
from scipy.interpolate import interp1d

fname = Path('1x3_PRISM_Q3_PSFLib.fits')

params = get_default_config()
params['instrument']['filter'] = "CLEAR"
params['instrument']['disperser'] = "PRISM"

psfs = PSFLib(fname.as_posix())
params["grid"]["x_grid"] = psfs.psf_x
params["grid"]["y_grid"] = psfs.psf_y
params["grid"]["wave_grid"] = psfs.psf_wave

model = Sersic(params)
model(params["grid"]["wave_grid"])
print(model.data.shape)

spec = Spec2D(params)
spec.make_spec2d(model,psfs)

params_2 = deepcopy(params)
params_2["geometry"]["shutter_j"] = params["geometry"]["shutter_j"] - 1
params_2["geometry"]["source_shutter"] = -1
spec_2 = Spec2D(params_2)
spec_2.make_spec2d(model, psfs)

params_3 = deepcopy(params)
params_3["geometry"]["shutter_j"] = params["geometry"]["shutter_j"] + 1
params_3["geometry"]["source_shutter"] = 1
spec_3 = Spec2D(params_3)
spec_3.make_spec2d(model, psfs)

lsf = LSF([spec, spec_2, spec_3], spec.wl_fpa, spec.x_fpa)
fit_output = lsf.compute_lsf()

disp_curve_stsci = Table.read(f'./data/jwst_nirspec_{params["instrument"]["disperser"].lower()}_disp.fits')
fdisp = interp1d(disp_curve_stsci['WAVELENGTH'],disp_curve_stsci['R'],kind='cubic')
w_prism = np.arange(0.5,6.0,0.05)
R_prism = fdisp(w_prism)

fig,ax = plt.subplots(1,2,figsize=(8,3.5))

wave_arr = w_prism
res_power_new = lsf.resolution(w_prism*1e4)
np.savetxt('msafits-Ferruit2022-defaults_PRISM-Rcurve-triple_shutter.txt',np.c_[wave_arr, res_power_new], fmt='%.4f', delimiter=',', header='wavelength_microns,R')

ax[0].plot(wave_arr, res_power_new, color='C0',ls='--',label='point source')
ax[0].plot(w_prism, R_prism,color='k',ls='solid',label='nominal resolution')
ax[0].set_xlabel(r"wavelength [micron]",fontsize=12)
ax[0].set_ylabel(r"resolution",fontsize=12)
ax[0].set_ylim(0,500)
ax[0].xaxis.set_minor_locator(AutoMinorLocator())
ax[0].yaxis.set_minor_locator(AutoMinorLocator())
ax[0].tick_params(direction='in',which='both',top=True,right=True)
ax[0].locator_params(nbins=6)

# ax[1].plot(wrange*1e-4,2*np.sqrt(2*np.log(2)) *lsf.dispersion_kms(wrange),color='C0',ls='--',label='point source')
# ax[1].plot(wrange*1e-4,3e5/(fdisp(wrange*1e-4)),color='k',ls='solid',label='nominal resolution')
# ax[1].set_xlabel(r"wavelength [micron]",fontsize=12)
# ax[1].set_ylabel(r"FWHM [$\rm km\,s^{-1}$]",fontsize=12)
# ax[1].xaxis.set_minor_locator(AutoMinorLocator())
# ax[1].yaxis.set_minor_locator(AutoMinorLocator())
# ax[1].tick_params(direction='in',which='both',top=True,right=True)
# ax[1].locator_params(nbins=6)
# ax[1].legend(fontsize=9)

plt.tight_layout()
plt.show()
