#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:25:51 2022

@author: yingshuyang
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time

from src.Material import Material
from src.TMM import SpecialMatrix
from src import exp_pulse, exp_tr, fourier, tools


def sim_pulse(layers, to_find, to_fit, omega, eps0, mu, d, f_in, unknown):
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
    t_0s        = TMM.transmission_coeff(T_0s)
    t_sinf      = TMM.transmission_coeff(T_sinf)
    t = TMM.transmission_coeff(T_0inf)
    t_noecho    = np.multiply(t_0s, t_sinf)
    # r           = TMM.reflection_coeff(T_0inf)
    r_noecho      = TMM.reflection_coeff(T_0s)
    # r_sinf      = TMM.reflection_coeff(T_sinf)
    # r_0s        = TMM.reflection_coeff(T_0s)

    '''Transmitted wave in freq domain'''
    f_S_R                 = f_in * t_0s
    f_inf_R_echofree      = f_in * t_noecho
    f_inf_R_echowith = f_in * t
    trans_echofree = fourier.ift(f_inf_R_echofree)
    trans_echowith = fourier.ift(f_inf_R_echowith)

    if to_fit[0] == True:
        return trans_echofree, f_inf_R_echofree
    else:
        return np.hstack((100*abs(t_noecho)**2, 100*abs(r_noecho)**2))


def minimize_func(layers, to_find, to_fit, omega, eps0, mu, d, f_in, f_out, transref, unknown):
    if to_find[0] == True:
        unknown = unknown[:len(unknown)//2] + 1j*unknown[len(unknown)//2:]
    if to_find[2] == True:
        unknown = np.array(np.array_split(unknown, 2))
    if to_fit[0] == True:
        sim_f_out = sim_pulse(layers, to_find, to_fit, omega, eps0, mu, d, f_in, unknown)[1]
        return np.sum(np.abs(sim_f_out - f_out))
    else:
        sim_transref = sim_pulse(layers, to_find, to_fit, omega, eps0, mu, d, f_in, unknown)
        return np.sum(np.abs(sim_transref - transref))


if __name__ == '__main__':
    f = open('/Users/yingshuyang/pythonfiles/TransferMatrixMethod/A_YIG_Fitting/config.json')
    config = json.load(f)
    to_fit = list(config['to_fit'].values())
    to_find = list(config['to_find'].values()) 
    input = list(config['input'].values())
    mat_data = config['layers']

    '''Material'''
    mu = 12.57e-7
    eps0 = 8.85e-12
    d = [i['thickness'] for i in mat_data]
    eps_data = [i['eps_data'] for i in mat_data]
    is_known = [i['is_known'] for i in mat_data]

    '''Experimental data'''
    if to_fit[0] == True:
        pulse_res = config['pulse_res']
        n, tmin, tmax, tpos = pulse_res.values()
        pulse_path = input[0]
        exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]
        exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1]
        t_grid, e_in, e_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n)
        omega, f_in = fourier.ft(t_grid, e_in)
        f_out = fourier.ft(t_grid, e_out)[1]
        transref = 1
    else:
        freq_range = config['freq_range']
        n, ev_range = freq_range.values()
        ev = np.linspace(ev_range[0], ev_range[1], n)
        omega = ev*2.42e14*6.28
        f_in = 1
        f_out = 1
        tr_path = input[1]
        trans, ref = exp_tr.read_tr(tr_path, omega)
        transref = np.hstack((trans, ref))

    '''Permittivity'''
    layers = []
    for j, k in zip(eps_data, is_known):
        if k == True:     # if known, find eps
            if type(j[0]) == str:
                layers.append(Material(omega).known_nk(j[0], j[1])*eps0)
            else:
                drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                layers.append(drude*eps0)
        else:
            layers.append('unknown')
            if to_find[0] == True:     # permittivities
                if type(j[0]) == str:
                    unknown = Material(omega).known_nk(j[0], j[1])
                else:
                    drude = Material(omega).drude(j[0], j[1]);     drude = 1*drude;  drude[0] = drude[1]
                    unknown = drude
            if to_find[1] == True:     # plasma and damping 
                unknown = np.array(j)
            if to_find[2] == True:     # n and k
                unknown = Material(omega).read_nk(j[0], j[1])
    
    # for transmission & reflection
    # unknown = (np.real(unknown)+1.5) - 1j*(np.imag(unknown)+2e-5)

    # #-----------------------------------------------------------------#
    min_sim_pulse = partial(sim_pulse, layers, to_find, to_fit, omega, eps0, mu, d, f_in)

    if to_fit[0] == True:
        sim_e_out, sim_f_out = min_sim_pulse(unknown)
    else: 
        sim_transref = min_sim_pulse(unknown)
        sim_trans = sim_transref[:len(sim_transref)//2]; sim_ref = sim_transref[len(sim_transref)//2:]

    #-----------------------------------------------------------------#
    if to_find[0] == True:       # perm
        new_unknown = np.hstack((np.real(unknown), np.imag(unknown))) 
    if to_find[1] == True:       #drude
        new_unknown = unknown
    if to_find[2] == True:       # n and k
        new_unknown = np.hstack((unknown[0], unknown[1]))

    min_func = partial(minimize_func, layers, to_find, to_fit, omega, eps0, mu, d, f_in, f_out, transref)
    if to_find[1] == True:
        bounds = Bounds(unknown*0.1, unknown*1.5)
    else:
        bounds = None
    start = time.time()
    res = minimize(min_func, new_unknown, method='Powell', bounds=bounds, options= {'disp' : True, 'adaptive': True, 'maxiter': 100000, 'maxfev': 100000})
    # res = minimize(min_func, new_unknown, method='L-BFGS-B')    #L-BFGS-B
    end = time.time()

    print(f'Elapsed time: {end - start}s')
    print(f'Before: {min_func(new_unknown)}')
    if to_find[0] == True:    
        result = res['x'][:len(new_unknown)//2] + 1j*res['x'][len(new_unknown)//2:]
        print(f'After: {min_func(np.hstack((np.real(result), np.imag(result))))}')
        
        ##==============yingshu testing  permittivtiy============         
        # drudePt =Material(omega).drude(5.145, 69.2e-3)
        # plt.figure('Permittivity')        
        # plt.plot(omega,drudePt.real,label = 'input real')
        # plt.plot(omega,drudePt.imag,label = 'input imag')   
        
        
        # permi = Material(omega).known_nk("SiO2.txt", "eV")      
        #plt.figure('Permittivity')     
        #plt.plot(omega,unknown.real,label = 'input real per')
        #plt.plot(omega,unknown.imag,label = 'input imag per')  
        
        #plt.plot(omega,result.real,label = 'output real per')
        #plt.plot(omega,result.imag,label = 'output imag per')
        #plt.legend()
        ##==========================================        
    if to_find[1] == True:
        result = res['x']
        print(f'After: {min_func(result)}')
        
        ##==============yingshu testing   Plasma damping============         
        drudePt = Material(omega).drude(5.145, 69.2e-3)
        drudePt_fit = Material(omega).drude(result[0], result[1])
        plt.figure('Plasma_damping')        
        plt.plot(omega,drudePt.real,label = 'input real drude')
        plt.plot(omega,drudePt.imag,label = 'input imag drude')        
        plt.plot(omega,drudePt_fit.real,label = 'output real drude')
        plt.plot(omega,drudePt_fit.imag,label = 'output imag drude')
        plt.legend()
        ##==========================================     
    if to_find[2] == True:
        result = np.array(np.array_split(res['x'], 2))
        freq = omega/(2*np.pi)
        n = result[0].real
        k = result[1].real
        print(f'After: {min_func(np.hstack((result[0], result[1])))}')
        ##==============yingshu testing   n and k============ 
        # omega = omega/6.28
        n_k = Material(omega).read_nk("YIG3.txt", "THz")      
        plt.figure('n_k')     
        plt.plot(freq,n_k[0],label = 'input n')
        plt.plot(freq,n_k[1],label = 'input k')     
        plt.plot(freq,result[0].real,label = 'output n')
        plt.plot(freq,result[1].real,label = 'output k')
        plt.legend()
    ##========================================== 
    print(f'Result: {result}')
    
    if to_fit[0] == True:
        omega = omega/6.28
        new_e_out, new_f_out = min_sim_pulse(result)
        freq = omega/(2*np.pi)
        A_permi = result
        '''Minimization plot - time'''
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))
        fig.suptitle('Minimized - time domain')
        ax1.set_title('Before')
        ax2.set_title('After')
        ax1.plot(t_grid, e_in, alpha=0.4,label = 'Air_exp')
        ax1.plot(t_grid, e_out,label = 'Sample_exp')
        ax1.plot(t_grid, sim_e_out,label = 'simulate_initialvalue')
        ax2.plot(t_grid, e_in, alpha=0.4,label = 'Air_exp')
        ax2.plot(t_grid, e_out, label='exp')
        ax2.plot(t_grid, new_e_out, label='simulate_afterfitting')
        ax1.legend()
        ax2.legend()
        plt.show()

        '''Minimization plot - freq'''
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(8, 4))
        fig.suptitle('Minimized - freq')
        ax1.set_title('Before')
        ax2.set_title('After')
        ax1.plot(freq, np.abs(f_out),label = 'Exp')
        ax1.plot(freq, np.abs(sim_f_out),label = 'Simulate_initialvalue')
        ax2.plot(freq, np.abs(f_out), label='Exp')
        ax2.plot(freq, np.abs(new_f_out), label='Simulate_afterFitting')
        ax2.legend()
        ax1.legend()
        plt.show()
    
    else:
        new_transref = min_sim_pulse(result)

        '''Minimization plot - trans'''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Minimized - trans')
        ax1.set_title('Before')
        ax2.set_title('After')
        ax1.plot(ev, trans)
        ax1.plot(ev, sim_trans)
        # ax1.set_ylim(0, 100)
        ax2.plot(ev, trans, label='exp')
        ax2.plot(ev, new_transref[:len(new_transref)//2], label='sim')
        # ax2.set_ylim(0, 100)
        ax2.legend()
        plt.show()

        '''Minimization plot - ref'''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Minimized - ref')
        ax1.set_title('Before')
        ax2.set_title('After')
        ax1.plot(ev, ref)
        ax1.plot(ev, sim_ref)
        # ax1.set_ylim(0, 100)
        ax2.plot(ev, ref, label='exp')
        ax2.plot(ev, new_transref[len(new_transref)//2:], label='sim')
        # ax2.set_ylim(0, 100)
        ax2.legend()
        plt.show()











