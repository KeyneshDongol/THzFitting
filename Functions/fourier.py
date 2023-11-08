import numpy as np


# def ft(t_grid, field_time):
#     dt = t_grid[1]-t_grid[0]
#     freq = np.fft.rfftfreq(len(t_grid), dt)
#     omega = 2.*np.pi*freq
#     f = np.fft.rfft(field_time)
#     field_freq = np.conjugate(f)
#     return omega, field_freq


# def ift(field_freq):
#     f = np.conjugate(field_freq)
#     field_time = np.fft.irfft(f)
#     return field_time

def ft(time,E_t):        
    delta_t = np.max(time)-np.min(time)
    last = round(len(time)/2)
    vec = np.arange(last+1)
    freqs = vec/(delta_t)
    omega = 2*np.pi*freqs
    # freqs = np.fft.fftfreq(len(time), delta_t)
    # omega = 2.*np.pi*freqs
    E_o = np.fft.rfft(E_t)
    E_onew = E_o/len(time)
    FT_E = E_onew[0:last+1]
    FT=FT_E
    FT[1:last+1]=2*FT[1:last+1]
    FT = np.conjugate(FT)   
    return omega,FT
    



def ift(field_freq):
    field_fre = np.conjugate(field_freq)
    f = np.fft.irfft(field_fre)*len(field_fre)
    return f