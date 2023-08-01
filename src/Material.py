import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d


class Material:
    def __init__(self, omega):
        self.omega = omega

    def read_nk(self, path, unit):
        nk_data = np.loadtxt(Path.cwd()/'material_data'/path)
        freq = nk_data[:,0]
        nindex = nk_data[:,1]
        kindex = nk_data[:,2]
        if unit == 'THz':
            freq = np.array(freq)*1e12*6.28              # angular frequency
        elif unit == 'eV':
            freq = np.array(freq)*2.42e14*6.28           # angular frequency
        f = interp1d(freq, nindex)
        h = interp1d(freq, kindex)
        new_n = f(self.omega)
        new_k = h(self.omega)
        return np.array([new_n, new_k])

    def epsilon(self, nk_data):
        nindex, kindex = nk_data
        eps = nindex**2 - kindex**2 + 2j*nindex*kindex
        return eps

    def known_nk(self, path, unit):
        nk_data = self.read_nk(path, unit)
        eps = self.epsilon(nk_data)
        return eps

    def drude(self, plasma, damping):
        eV2Hz = 241.79893e12                           # 1eV = 241.79893 THz
        plas = plasma*eV2Hz*2*np.pi
        damp = damping*eV2Hz*2*np.pi
        epsilon = 1 - plas**2/(self.omega**2 + 1j*damp*self.omega)
        return epsilon

    def known_eps(self, path):
        eps_data = np.loadtxt(Path.cwd()/'material_data'/path)
        freq = eps_data[:,0]
        eps_real = eps_data[:,1]
        eps_imag = eps_data[:,2]
        eps = eps_real + 1j*eps_imag
        espline = interp1d(freq, eps)
        eps = espline(self.omega)
        return eps

