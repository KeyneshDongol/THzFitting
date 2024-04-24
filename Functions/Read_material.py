import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.constants import epsilon_0 as eps0



class Material(object):
    def __init__(self, omega):
        self.omega = omega
        self.layers = []  # Initialize the empty list for storing layer properties
    
    def _check_omega_within_range(self, freq):
        '''
        Check if the angular frequency omega is within the range of the provided frequency data
        '''
        min_freq = np.min(freq)
        max_freq = np.max(freq)
        if not (min_freq <= self.omega).all() or not (self.omega <= max_freq).all():
            raise ValueError(f"Omega values are out of the allowable range ({min_freq}, {max_freq})")

    # def add_layer(self, material_data, is_known):
    #     if is_known:
    #         if isinstance(material_data[0], str):  # Path to nk data
    #             eps = self.known_nk(*material_data) * eps0
    #         else:  # Drude parameters
    #             eps = self.drude(*material_data) * eps0
    #     else:
    #         eps = 'unknown'
    #     self.layers.append(eps)
    
    def read_nk(self, path, unit):
        nk_data = np.loadtxt(Path.cwd()/'material_data'/path)
        freq = nk_data[:,0]
        nindex = nk_data[:,1]
        kindex = nk_data[:,2]
        if unit == 'THz':
            freq = np.array(freq)*1e12*6.28              # angular frequency
        elif unit == 'eV':
            freq = np.array(freq)*2.42e14*6.28           # angular frequency
        
        self._check_omega_within_range(freq)  # Check if omega is within the data range

        f = interp1d(freq, nindex) 
        h = interp1d(freq, kindex)
        return np.array([f(self.omega), h(self.omega)])

    def epsilon(self, nk_data):
        nindex, kindex = nk_data
        return nindex**2 - kindex**2 + 2j*nindex*kindex

    def known_nk(self, path, unit):
        nk_data = self.read_nk(path, unit)
        return self.epsilon(nk_data)

    # def drude(self, plasma, damping):
    #     eV2Hz = 241.79893e12
    #     plas = plasma * eV2Hz * 2 * np.pi
    #     damp = damping * eV2Hz * 2 * np.pi
    #     return 1 - plas**2 / (self.omega**2 + 1j * damp * self.omega)
    def drude(self, plasma_freq, damping_freq):
        print(f"Processing Drude model with plasma_freq: {plasma_freq}, damping_freq: {damping_freq}")
        try:
            eV2Hz = 241.79893e12

            plas = plasma_freq * eV2Hz * 2 * np.pi
            return 1 - (plas ** 2) / (self.omega ** 2 + 1j * self.omega * damping_freq)
        except TypeError as e:
            print(f"Data type error in drude calculation: {e}")
            raise

    def known_eps(self, path):
        eps_data = np.loadtxt(Path.cwd()/'material_data'/path)
        freq = eps_data[:,0]
        eps_real = eps_data[:,1]
        eps_imag = eps_data[:,2]
        self._check_omega_within_range(freq)  # Check if omega is within the data range
        eps = eps_real + 1j*eps_imag
        espline = interp1d(freq, eps)
        return espline(self.omega)
    
    # def get_layer_properties(self):
    #     '''
    #     Allows us to access the properties of all layers at once
    #     '''
    #     return self.layers
