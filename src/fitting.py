import numpy as np

import Material, fourier
from TMM import SpecialMatrix


def sim_pulse(layers, to_find, omega, eps0, mu, d, f_in, unknown):
    epsilon = []
    for l in layers:
        if type(l) == str:  # unknown
            if to_find[0] == True:                                        # permittivity
                epsilon.append(unknown*eps0)
            if to_find[1] == True:                                        # plasma and damping
                eps = Material(omega).drude(unknown[0], unknown[1]);  eps= 1*eps;  eps[0] = eps[1]
                epsilon.append(eps*eps0)
            if to_find[2] == True:                                        # n and k
                epsilon.append(Material(omega).epsilon(unknown)*eps0)
        else:
            epsilon.append(l)

    '''Transfer matrix'''
    TMM = SpecialMatrix(omega, mu, epsilon, d)
    T_0s, T_sinf, T_0inf = TMM.transfer_matrix()

    '''Tranmission & reflection coefficients'''
    # t_0s        = TMM.transmission_coeff(T_0s)
    # t_sinf      = TMM.transmission_coeff(T_sinf)
    t = TMM.transmission_coeff(T_0inf)
    # t_noecho    = np.multiply(t_0s, t_sinf)
    # r           = TMM.reflection_coeff(T_0inf)
    # r_sinf      = TMM.reflection_coeff(T_sinf)
    # r_0s        = TMM.reflection_coeff(T_0s)

    '''Transmitted wave in freq domain'''
    # f_S_R                 = f_in * t_0s
    # f_inf_R_echofree      = f_in * t_noecho
    f_inf_R_echowith = f_in * t
    # trans_echofree = fourier.ift(f_inf_R_echofree)
    trans_echowith = fourier.ift(f_inf_R_echowith)

    return trans_echowith, f_inf_R_echowith, t

def minimize_func(layers, to_find, omega, eps0, mu, d, f_in, f_out, unknown):
    if to_find[0] == True:
        unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
    if to_find[2] == True:
        unknown = np.array(np.array_split(unknown, 2))
    return np.sum(np.abs(sim_pulse(layers, to_find, omega, eps0, mu, d, f_in, unknown)[1] - f_out))
