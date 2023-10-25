#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:10:27 2023

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
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

