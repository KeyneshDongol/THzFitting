import numpy as np


def ft(t_grid, field_time):
    dt = t_grid[1]-t_grid[0]
    freq = np.fft.rfftfreq(len(t_grid), dt)
    omega = 2.*np.pi*freq
    f = np.fft.rfft(field_time)
    field_freq = np.conjugate(f)
    return omega, field_freq


def ift(field_freq):
    f = np.conjugate(field_freq)
    field_time = np.fft.irfft(f)
    return field_time

