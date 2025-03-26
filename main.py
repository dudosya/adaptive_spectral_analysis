from signal_generation import sine_additions  
from plot_funcs import *
from lms import *
import matplotlib.pyplot as plt
import numpy as np

import time

# config
start_time = 0
end_time = 0.5
sampling_rate = 1000
amplitude_list = [1,1,1,1]
frequency_list = [100, 200, 300, 400]
phase_list = [0, 30, 60, 0]

filter_order = 50  # Example: Try different values (e.g., 16, 64)
step_size = 0.001 # Example: Adjust for convergence (e.g., 0.001, 0.05)  Critical parameter
delay = 1     # Typically 1

# running funcs
signal = sine_additions(start_time=start_time, end_time=end_time, sampling_rate=sampling_rate,amplitude_list=amplitude_list , frequency_list=frequency_list, phase_list=phase_list)
#plot_signal(signal, sampling_rate)

#fft spectrum analysis

start_time = time.perf_counter()

fft_output = np.fft.fft(signal)

end_time = time.perf_counter()
exec_time_FFT = end_time-start_time
print(f"FFT Exec time: {exec_time_FFT:.15f}")

plot_spectrum(signal,fft_output,sampling_rate)

#adaptive spectrum analysis
start_time = time.perf_counter()

converged_weights, error_signal, y_output = lms_filter(signal, filter_order, step_size, delay)
frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)


end_time = time.perf_counter()
exec_time_LMS = end_time-start_time
print(f"LMS Exec time: {exec_time_LMS:.15f}")
print(f"FFT is {exec_time_LMS/exec_time_FFT:.1f} times faster than LMS")


plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output)

