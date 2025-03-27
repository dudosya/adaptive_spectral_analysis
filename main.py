from signal_generation import *  
from plot_funcs import *
from lms import *

import matplotlib.pyplot as plt
import numpy as np
import time

# signal config
start_time = 0
end_time = 0.1
sampling_rate = 1000

sine_dict = {"type":"sine", "freq": 100, "amp": 0.5, "phase": 0}
am_dict = {"type":"am", "freq": 300, "envelope": None, "phase": 0}
chirp_dict = {'type': 'chirp', 'start_freq': 1, 'end_freq': 10, 'amp': 1}

signals_list = [sine_dict]
#signals_list = [sine_dict,am_dict,chirp_dict]

# filter config
filter_order = 50  
step_size = 0.05 
delay = 1     

# running funcs
time_array = generate_time_vector(start_time, end_time, sampling_rate)
signal = sum_signal_components(time_array, signals_list )
plot_signal(signal, sampling_rate)

#fft spectrum analysis

start_timer = time.perf_counter()

fft_output = np.fft.fft(signal)

end_timer = time.perf_counter()
exec_time_FFT = end_timer-start_timer
print(f"FFT Exec time: {exec_time_FFT:.15f}")

plot_spectrum(signal,fft_output,sampling_rate)

# #adaptive spectrum analysis
start_timer = time.perf_counter()

converged_weights, error_signal, y_output = lms_filter(signal, filter_order, step_size, delay)
frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)


end_timer = time.perf_counter()
exec_time_LMS = end_timer-start_timer
print(f"LMS Exec time: {exec_time_LMS:.15f}")
print(f"FFT is {exec_time_LMS/exec_time_FFT:.1f} times faster than LMS")


plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output)

