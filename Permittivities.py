from io import TextIOWrapper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from math import pi
from matplotlib.gridspec import GridSpec
from numpy._typing import _128Bit
from numpy.typing import NDArray
from Functions.Read_material import Material
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize
from Functions.mainfunction import E_TMM, Error_func,  noise_remove
from toIntroduce import E_AirIn

class Permittivities:

    start: TextIOWrapper = open(Path.cwd()/'inputs.json')
    config = json.load(start)
    toFind: list = list(config['to_find'].values())
    _mu: float = 12.57e-7
    _eps0: float = 8.85e-12

    def __init__(self, omega, epsData, isKnown):
        self.omega = omega
        self.epsData = epsData
        self.isKnown = isKnown



    def permittivities(self) -> None:
        layers = []
        j: list
        k: bool
        for j, k in zip(self.epsData, self.isKnown):
            if k == True: # if known -> find eps
                if type(j[0]) == str
                    layers.append(Material(self.omega).known_nk(j[0], j[1]) * self._eps0)
                else:
                    drude = Material(self.omega).drude(j[0], j[1]); drude = 1*drude; drude[0] = drude[1]
                    layers.append(drude * self._eps0)
            else:
                layers.append('unknown')
                if self.toFind[0] == True: # permittivity (real and imaginary)
                    if type(j[0]) == str:
                        unknown = Material(self.omega).known_nk(j[0], j[1])
                    else:
                        drude = Material(self.omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                        unknown = drude
                if self.toFind[1] == True: # plasma and damping
                    unknown = np.array(j)
                if self.toFind[2] == True
                    unknown = Material(self.omega).read_nk(j[0], j[1])
