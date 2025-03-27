from signal_generation import *  
from plot_funcs import *
from lms import *
from rls import *
from utils import *

import matplotlib.pyplot as plt
import numpy as np


# signal config
start_time = 0
end_time = 0.02
sampling_rate = 1000

sine_dict = {"type":"sine", "freq": 100, "amp": 0.5, "phase": 0}
am_dict = {"type":"am", "freq": 200, "envelope": None, "phase": 0}
chirp_dict = {'type': 'chirp', 'start_freq': 300, 'end_freq': 400, 'amp': 0.5}
noise_dict = {'type': 'noise', 'mean': 0.0, 'std_dev': 0.01} 

signals_list = [sine_dict, am_dict, chirp_dict, noise_dict]

time_array = generate_time_vector(start_time, end_time, sampling_rate)
signal = sum_signal_components(time_array, signals_list )

# plot_signal(signal, sampling_rate)
# plot_signal_components(time_array, signals_list)




#fft spectrum analysis
fft_output = np.fft.fft(signal)
#plot_spectrum(signal,fft_output,sampling_rate)




# # #adaptive spectrum analysis

# NLMS Experiments
# Experiment with filter order
# NLMS_filter_order_list = [2,4,8,16,32,64]
# step_size = 0.02
# delay = 1
# reg_factor=1e-6

# for filter_order in NLMS_filter_order_list:
#     converged_weights, error_signal, y_output = nlms_filter(signal, filter_order, step_size, delay, reg_factor)
#     frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)
#     plot_filter_results(error_signal, frequencies, psd, signal, sampling_rate, y_output, "NLMS")

#Experiment with delay
# NLMS_filter_order = 32
# step_size = 0.05
# delay_list = [1,2,4,8,16]
# reg_factor=1e-6

# for delay in delay_list:
#     converged_weights, error_signal, y_output = nlms_filter(signal, NLMS_filter_order, step_size, delay, reg_factor)
#     frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)
#     plot_filter_results(error_signal, frequencies, psd, signal, sampling_rate, y_output, "NLMS")




# # NLMS config
# NLMS_filter_order = 50
# step_size = 0.01
# delay = 1
# reg_factor=1e-6


# converged_weights, error_signal, y_output = nlms_filter(signal, NLMS_filter_order, step_size, delay, reg_factor)
# frequencies, psd = adaptive_spectrum(converged_weights, sampling_rate)
# plot_filter_results(error_signal, frequencies, psd, signal, sampling_rate, y_output, "NLMS")


# #RLS config

# RLS experiment with filter order
RLS_filter_order_list = [2,4,8,16,32,64]
step_size = 0.001
delay = 1     
forgetting_factor_rls = 0.995  # Typical value, closer to 1 means more memory
delta_rls = 1.0 


for RLS_filter_order in RLS_filter_order_list:
    weights_rls, error_rls, y_rls = rls_filter(
    signal, RLS_filter_order, forgetting_factor_rls, delay, delta_rls
    )
    frequencies_rls, psd_rls = adaptive_spectrum(weights_rls, sampling_rate)
    plot_filter_results(error_rls, frequencies_rls, psd_rls, signal, sampling_rate, y_rls, "RLS")












# RLS_filter_order = 50
# step_size = 0.001
# delay = 1     
# forgetting_factor_rls = 0.995  # Typical value, closer to 1 means more memory
# delta_rls = 1.0 


# weights_rls, error_rls, y_rls = rls_filter(
#     signal, RLS_filter_order, forgetting_factor_rls, delay, delta_rls
# )
# frequencies_rls, psd_rls = adaptive_spectrum(weights_rls, sampling_rate)
# plot_filter_results(error_rls, frequencies_rls, psd_rls, signal, sampling_rate, y_rls, "RLS")
