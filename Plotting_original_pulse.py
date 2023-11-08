#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:43:18 2023

@author: yingshuyang
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from pathlib import Path
import matplotlib.pyplot as plt
from functools import partial
import json
import time
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from math import pi
from matplotlib.gridspec import GridSpec
from Functions.Read_material import Material
from Functions.TMM import SpecialMatrix
from Functions import exp_pulse, fourier, tools
from scipy.optimize import minimize, LinearConstraint
from Functions.mainfunction import E_TMM, Error_func,  noise_remove



'''inputs'''
f = open(Path.cwd()/'inputs1.json')
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
t_grid, E_air_in= exp_pulse.read_pulse(exp_in_pulse) 
omega, E_sample_out= exp_pulse.read_pulse(exp_out_pulse) 
t_grid_new, E_air_in_new, E_sample_out_new = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, tmin, tmax, tpos, d, n) # data reading (propagated through air)




# =============================================================================
# Plottings of raw data
# =============================================================================

# Convert time grid for plotting
t_grid_plot = t_grid  # convert to ps for plotting
t_grid_new_plot = t_grid_new * 1e12  # convert to ps for plotting

# Define the style settings
figsize = (6, 6)  # figure size in inches
linewidth = 2  # line width for the plots
tick_labelsize = 10  # size of the ticks labels
label_fontsize = 10  # font size for the x and y labels
title_fontsize = 10  # font size for the titles
colors = ['blue', 'green']  # colors for the plots

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

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

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure if needed
plt.savefig('processed_data_plots.png', dpi=300)

# Show the plot
plt.show()











