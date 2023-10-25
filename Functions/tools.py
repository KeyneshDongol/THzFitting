import numpy as np
from pathlib import Path


def save_eps(path, omega, eps):
    eps_real = np.real(eps)
    eps_imag = np.imag(eps)
    eps_data = Path.cwd()/'material_data'/path
    with open(eps_data, 'w') as f:
        for freq, real, imag in zip(omega, eps_real, eps_imag):
            f.write(str(freq)+' '+str(real)+' '+str(imag)+'\n')

