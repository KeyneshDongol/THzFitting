import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def read_tr(path, omega):
    tr_data = np.loadtxt(Path.cwd()/'experimental_data'/path)
    freq = ((2.99792458e8/(tr_data[:,0]*1e-9))/1e12)*0.00414
    freq = freq*2.42e14*6.28
    trans = tr_data[:,1]
    ref = tr_data[:,2]
    f = interp1d(freq, trans)
    h = interp1d(freq, ref)
    new_trans = f(omega)
    new_ref = h(omega)
    return new_trans, new_ref
