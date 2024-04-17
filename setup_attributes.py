import json
import numpy as np
from pathlib import Path
from Functions import exp_pulse, fourier, tools


class ToFind:
    def __init__(self, permittivity = False, plasma_damping = False, n_k = False):
        self.permittivity = permittivity
        self.plasma_damping = plasma_damping
        self.n_k = n_k

class Input:
    def __init__(self, pulse = None):
        if pulse is None:
            raise ValueError("Pulse must be provided.")
        self.pulse = pulse

class PulseRes:
    def __init__(self, n = 0, tmin = 0, tmax = 0, tpos = 0):
        self.n = n
        self.tmin = tmin
        self.tmax = tmax
        self.tpos = tpos

class Layer:
    def __init__(self, thickness = [], eps_data = None, is_substrate = False, is_known = True):
        if eps_data is None:
            raise ValueError("Error: No eps_data was provided.")
        self.thickness = thickness
        self.eps_data = eps_data
        self.is_substrate = is_substrate
        self.is_known = is_known

class Echoes:
    def __init__(self, Removed = False):
        self.Removed = Removed

class ExperimentalData:
    def __init__(self, config, base_path='experimental_data'):
        self.config = config
        self.base_path = Path.cwd() / base_path
        pulse_res = config.get('pulse_res', {})
        self.n, self.tmin, self.tmax, self.tpos = pulse_res.values()

        input_data = config.get('input', {})
        self.pulse_path = [self.base_path / path for path in input_data.get('pulse', [])]

    def process_experimental_data(self, d, n=None):
        if len(self.pulse_path) < 2:
            raise ValueError("Insufficient pulse paths provided")

        exp_in_pulse, exp_out_pulse = self.pulse_path
        t_grid, E_air_in, E_sample_out = exp_pulse.fitted_pulse(exp_in_pulse, exp_out_pulse, self.tmin, self.tmax, self.tpos, d, n or self.n)
        omega, E_air_f = fourier.ft(t_grid, E_air_in)
        _, E_exp_f = fourier.ft(t_grid, E_sample_out)

        self.t_grid = t_grid
        self.E_air_in = E_air_in
        self.E_sample_out = E_sample_out
        self.omega = omega
        self.E_air_f = E_air_f
        self.E_exp_f = E_exp_f

class Attributes:
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as file:
            config = json.load(file)

        required_sections = ['to_find', 'input', 'pulse_res', 'layers', 'Echoes']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        #TODO: Maybe include dafualt txt files for future uses?

        # Initialize constants
        self.mu = 12.57e-7
        self.eps0 = 8.85e-12

        # Initialize JSON data
        self.to_find = ToFind(**config.get('to_find', {}))
        self.input = Input(**config.get('input', {}))
        self.pulse_res = PulseRes(**config.get('pulse_res', {}))
        self.layers = [Layer(**layer) for layer in config.get('layers', [])]
        self.echoes = Echoes(**config.get('Echoes', {}))

        # Initialize material data
        self.process_material_data()

        # Pass the whole config or just the relevant part to ExperimentalData
        self.experimental_data = ExperimentalData(config)

        # Initialize experimental data
        self.experimental_data.process_experimental_data(self.d)

    def process_material_data(self):
        # Process the layers data (e.g., extracting thicknesses and permittivity data)
        self.d = [layer.thickness for layer in self.layers]
        self.eps_data = [layer.eps_data for layer in self.layers]
        self.is_known = [layer.is_known for layer in self.layers]
        self.is_substrate = [layer.is_substrate for layer in self.layers]
        self.sub_layer = np.where(self.is_substrate)[0][0] if self.is_substrate else None
