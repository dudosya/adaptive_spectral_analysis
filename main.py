from signal_generation import sine_additions  
from plot_funcs import *
from lms import *
import matplotlib.pyplot as plt
import numpy as np

# config
start_time = 0
end_time = 0.2
sampling_rate = 1000
amplitude_list = [1,1,1,1]
frequency_list = [100, 200, 300, 400]
phase_list = [0, 30, 60, 0]

filter_order = 32  # Example: Try different values (e.g., 16, 64)
step_size = 0.01 # Example: Adjust for convergence (e.g., 0.001, 0.05)  Critical parameter
delay = 1      # Typically 1

# running funcs
signal = sine_additions(start_time=start_time, end_time=end_time, sampling_rate=sampling_rate,amplitude_list=amplitude_list , frequency_list=frequency_list, phase_list=phase_list)
plot_signal(signal, sampling_rate)

#fft spectrum analysis
fft_output = np.fft.fft(signal)
plot_spectrum(signal,fft_output,sampling_rate)


#adaptive spectrum analysis
converged_weights, error_signal, y_output = lms_filter(signal, filter_order, step_size, delay)
frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)
plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output)

