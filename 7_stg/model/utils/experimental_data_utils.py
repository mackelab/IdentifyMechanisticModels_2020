import numpy as np
from copy import deepcopy
import scipy.signal


def calc_spikes(trace, threshold=1.0, mode='local_max'):
    trace = np.asarray(trace)

    if mode=='local_max':
        trace_above_th = find_local_max(trace, threshold)
    else:
        trace_above_th = trace > threshold
        trace_above_th = replace_1bySpike(trace_above_th)
    trace_above_th = np.asarray(trace_above_th, dtype=int)

    return trace_above_th


def find_local_max(V, threshold):
    time_diff = 0.1 # ms
    wsize = int(0.5 / time_diff)
    voltage = np.zeros_like(V)
    V = scipy.signal.savgol_filter(V, wsize, 3)

    spike_inds = np.where((V[1:-1] > threshold) & (np.diff(V[:-1]) >= 0) & (np.diff(V[1:]) <= 0))[0]
    voltage[spike_inds] = 1.0
    return voltage

def replace_1bySpike(binary_trace):
    spike_list = np.zeros_like(binary_trace)

    current = 0
    counter = 0
    for num in binary_trace:
        if num:
            current += 1
        else:
            if current > 0:
                spike_list[int(counter - current/2)] = 1
            current = 0
        counter += 1
    return spike_list


def subtract_PDfromPY(PY_spikes, PD_spikes, vicinity=10, sampling_frequency=10000):
    counter = 0
    PYminusPD = deepcopy(PY_spikes)
    for s_PY in PY_spikes:
        PDinVicitity = False
        if s_PY > 0:
            PDinVicitity = check_spike_vicinity(counter, PD_spikes, vicinity=vicinity, sampling_frequency=sampling_frequency)
        if PDinVicitity:
            PYminusPD[counter] = 0.0
        counter += 1
    return PYminusPD


def check_spike_vicinity(counter, PD_spikes, vicinity=10, sampling_frequency=10000):
    if np.any(PD_spikes[int(counter-vicinity*sampling_frequency/1000):int(counter+vicinity*sampling_frequency/1000)]): return True
    else: return False