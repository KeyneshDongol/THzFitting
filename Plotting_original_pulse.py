#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:43:18 2023

@author: yingshuyang
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from Functions import exp_pulse, fourier
from math import pi



'''inputs'''
f = open(Path.cwd()/'inputs.json')
config = json.load(f)
to_find = list(config['to_find'].values()) 
input_num = list(config['input'].values())
mat_data = config['layers']
echoes_removed =  list(config['Echoes'].values())

'''Material and Geometry of the sample'''
mu = 12.57e-7
eps0 = 8.85e-12
d = [i['thickness'] for i in mat_data]       #list
eps_data = [i['eps_data'] for i in mat_data] #list 
is_known = [i['is_known'] for i in mat_data] #list
is_substrate = [i['is_substrate'] for i in mat_data] #list
sub_layer = np.where(is_substrate)[0][0]



pulse_res = config['pulse_res']  
n, tmin, tmax, tpos = pulse_res.values()
pulse_path = input_num[0]
exp_in_pulse = Path.cwd()/'experimental_data'/pulse_path[0]  # current working directory
exp_out_pulse = Path.cwd()/'experimental_data'/pulse_path[1] # current working directory

t_grid, E_air_in= exp_pulse.read_pulse(exp_in_pulse) # reading the original pulse E_ref
_, E_sample_out= exp_pulse.read_pulse(exp_out_pulse) # reanding the original pulse E_sample
omega_o, E_air_f_o = fourier.ft(t_grid*1e-12, E_air_in)
omega_o, E_sample_f_o = fourier.ft(t_grid*1e-12, E_sample_out)
freq_o = omega_o*1e-12/2*pi



t_grid_new, E_air_in_new, E_sample_out_new = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)
omega, E_air_f = fourier.ft(t_grid_new, E_air_in_new)
E_exp_f = fourier.ft(t_grid_new, E_sample_out_new)[1]

'''Transfer function'''
T_exp_f1 = E_exp_f/E_air_f


T_exp_f_o = E_sample_f_o/E_air_f_o







# =============================================================================
# Plottings of raw data
# =============================================================================
freq = omega*1e-12/2*pi



# plt.figure('test')
# # plt.plot(freq,np.abs(E_air_f),label = 'aire')
# # plt.plot(freq,np.abs(E_exp_f),label = 'sample')
# plt.plot(freq,np.abs(T_exp_f1),label = 'Trans func1')
# # plt.plot(freq_o,np.abs(T_exp_f_o),label = 'Trans func_o')
# plt.legend()




# Convert time grid for plotting
t_grid_plot = t_grid  # convert to ps for plotting
t_grid_new_plot = t_grid_new * 1e12  # convert to ps for plotting





# Plotting parameters
figsize = (8, 10)  # Adjusted for 3 subplots
linewidth = 2
colors = ['blue', 'red']
label_fontsize = 12
tick_labelsize = 10
title_fontsize = 14

# Create a figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

# First subplot for raw data
ax1.plot(t_grid_plot, E_air_in, label='E_air_in', linewidth=linewidth, color=colors[0])
ax1.plot(t_grid_plot, E_sample_out, label='E_sample_out', linewidth=linewidth, color=colors[1])
ax1.set_xlabel('Time (ps)', fontsize=label_fontsize)
ax1.set_ylabel('Electric Field', fontsize=label_fontsize)
ax1.tick_params(axis='both', labelsize=tick_labelsize)
ax1.legend(fontsize=label_fontsize)
ax1.set_title('Raw data', fontsize=title_fontsize)

# Second subplot for processed raw data
ax2.plot(t_grid_new_plot, E_air_in_new, label='E_air_in_processed', linewidth=linewidth, color=colors[0])
ax2.plot(t_grid_new_plot, E_sample_out_new, label='E_sample_out_processed', linewidth=linewidth, color=colors[1])
ax2.set_xlabel('Time (ps)', fontsize=label_fontsize)
ax2.set_ylabel('Electric Field', fontsize=label_fontsize)
ax2.tick_params(axis='both', labelsize=tick_labelsize)
ax2.legend(fontsize=label_fontsize)
ax2.set_title('Processed raw data', fontsize=title_fontsize)

# Third subplot for frequency domain
ax3.plot(freq, np.abs(E_air_f), label='E_in', color= colors[0])
ax3.plot(freq, np.abs(E_exp_f), label='E_out', color= colors[1])
ax3.set_xlabel('Freq.(THz)', fontsize=label_fontsize)
ax3.set_ylabel('E($\omega$) arb.units', fontsize=label_fontsize)
ax3.tick_params(axis='both', labelsize=tick_labelsize)
ax3.legend(fontsize=label_fontsize)
ax3.set_title('Freq. domain', fontsize=title_fontsize)
ax3.set_xlim(0, 50)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure if needed
plt.savefig('original_data_plots.png', dpi=200)

# Show the plot
plt.show()




