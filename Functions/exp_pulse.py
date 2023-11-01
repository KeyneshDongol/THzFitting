import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


from scipy.signal import find_peaks, savgol_filter

def read_pulse(path):
    t_grid = []
    e_t = []
    with open(path) as f:
        for line in f:
            data = line.split()
            t_grid.append(float(data[0]))
            e_t.append(float(data[1]))
    return np.array(t_grid), np.array(e_t)

def find_start(t_grid, e_in, e_out):

    diff_threshold = 0.0025
    
    start_idx = 0
    for i in range(len(e_in)):
        if e_in[i] > diff_threshold:
            start_idx = i
            break    
    return t_grid[start_idx-20:], e_in[start_idx-20:], e_out[start_idx-20:]

def fitted_pulse(path_in, path_out, tmin, tmax, tpos, d, n):
    t_grid, e_in = read_pulse(path_in)
    e_out = read_pulse(path_out)[1]
    t_grid, e_in, e_out = find_start(t_grid, e_in, e_out)
    

    t_0 = t_grid[0]
    t_in = [(t-t_0)*1e-12+tpos for t in t_grid]
    t_out = [(t-t_0)*1e-12+tpos+(np.sum(d)/2.99792458e8) for t in t_grid]            # add air propagation

    t_in = np.array(t_in); t_out = np.array(t_out); e_in = np.array(e_in); e_out = np.array(e_out)

    t_prolong1 = np.linspace(tmin, tpos, 50)
    t_prolong2 = np.linspace(t_grid[-1], tmax, 50)
    t_in = np.hstack((t_prolong1, t_in, t_prolong2))
    t_out = np.hstack((t_prolong1, t_out, t_prolong2))
    e_t_prolong = np.linspace(0, 0, 50)
    e_in = np.hstack((e_t_prolong, e_in, e_t_prolong))
    e_out = np.hstack((e_t_prolong, e_out, e_t_prolong))
    f = interp1d(t_in, e_in)
    h = interp1d(t_out, e_out)
    t_new = np.linspace(tmin, tmax, n)
    e_in_new = f(t_new)
    e_out_new = h(t_new)
    return t_new, e_in_new, e_out_new



