import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from signal_generation import *

def plot_signal(signal, sampling_rate, title="Signal"):
    signal = np.asarray(signal)
    time = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)
    print(f"The signal has {len(time)} samples")

    plt.figure()
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def plot_signal_components(time_array, component_list):
    """
    Generates and plots each signal component specified in component_list
    individually in separate plot windows.

    Args:
        time_array (np.ndarray): The time vector.
        component_list (list): A list where each element is a dictionary
                               describing a signal component (same format
                               as for sum_signal_components).
    """
    print(f"Plotting {len(component_list)} signal components individually...")
    signal_length = len(time_array)

    for i, comp in enumerate(component_list):
        comp_type = comp.get('type', 'sine').lower()
        component_signal = np.zeros_like(time_array) # Initialize for this component
        title_suffix = "" # To store specific parameters for the title

        try:
            if comp_type == 'sine':
                freq = comp.get('freq', 1)
                amp = comp.get('amp', 1.0)
                phase = comp.get('phase', 0)
                component_signal = sine_wave_generator(time_array, freq, amp, phase)
                title_suffix = f"(Freq: {freq} Hz, Amp: {amp})"
            elif comp_type == 'am':
                freq = comp.get('freq', 1)
                envelope = comp.get('envelope', 1.0) # Using 1.0 if None for plotting
                phase = comp.get('phase', 0)
                # Re-generate envelope if needed, handle different types for title
                if callable(envelope):
                     envelope_desc = "Function"
                elif isinstance(envelope, (np.ndarray, list)):
                     envelope_desc = f"Array[{len(envelope)}]"
                else:
                     envelope_desc = f"Scalar({envelope})"

                component_signal = am_sine_generator(time_array, freq, envelope, phase)
                title_suffix = f"(Carrier Freq: {freq} Hz, Envelope: {envelope_desc})"
            elif comp_type == 'chirp':
                start_freq = comp.get('start_freq', 1)
                end_freq = comp.get('end_freq', 1)
                amp = comp.get('amp', 1.0)
                phase = comp.get('phase', 0)
                component_signal = linear_chirp_generator(time_array, start_freq, end_freq, amp, phase)
                title_suffix = f"(Freq: {start_freq} Hz to {end_freq} Hz, Amp: {amp})"
            elif comp_type == 'noise':
                mean = comp.get('mean', 0.0)
                std_dev = comp.get('std_dev', 0.1)
                component_signal = gaussian_noise_generator(signal_length, mean, std_dev)
                title_suffix = f"(Mean: {mean}, Std Dev: {std_dev})"
            else:
                print(f"Skipping unknown component type '{comp_type}' for plotting.")
                continue # Skip to next component

            # --- Plotting the current component ---
            plt.figure(figsize=(10, 4)) # Create a new figure for each component
            plt.plot(time_array, component_signal)
            plt.title(f"Signal Component {i+1}: {comp_type.upper()} {title_suffix}")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.tight_layout()
            print(f"Showing plot for component {i+1} ({comp_type}). Close plot window to continue...")
            plt.show() # Display the plot and wait until it's closed

        except Exception as e:
            print(f"Error plotting component {i+1} ({comp_type}): {e}")
            # Optionally close any partially created figure if error occurs mid-plot
            plt.close() # Close potentially broken figure window

    print("Finished plotting all components.")




def plot_spectrum(signal, fft_output, sampling_rate = 1000, threshold_ratio=0.1, prominence=None):
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    magnitude = np.abs(fft_output)
    phase = np.angle(fft_output)

    positive_freq_indices = np.where(frequencies >= 0)[0]


    positive_frequencies = frequencies[positive_freq_indices]
    positive_magnitude = magnitude[positive_freq_indices]
    positive_phase = phase[positive_freq_indices]

    
    if len(positive_frequencies) == 0:
        print("Warning: No positive frequencies found for FFT spectrum analysis.")
        return # Cannot plot or find peaks


    # --- Peak Finding ---
    print("\n--- FFT Peak Frequencies ---")
    if len(positive_magnitude) > 0 and np.max(positive_magnitude) > 0:
        min_height = np.max(positive_magnitude) * threshold_ratio
        peak_indices, properties = find_peaks(positive_magnitude, height=min_height, prominence=prominence)
        peak_frequencies = positive_frequencies[peak_indices]

        if len(peak_frequencies) > 0:
            peak_frequencies.sort()
            print(f"Found {len(peak_frequencies)} peaks at frequencies (Hz):")
            print([f"{freq:.2f}" for freq in peak_frequencies])
            # Optionally print heights as well:
            # print(f"Peak Magnitudes (linear): {[f'{positive_magnitude[i]:.2f}' for i in peak_indices]}")
        else:
            print(f"No significant peaks found above threshold ({threshold_ratio*100:.1f}% of max) or with specified prominence.")
            print(f"Max Magnitude value: {np.max(positive_magnitude):.4f}")

    elif len(positive_magnitude) > 0:
         print("Magnitudes are all zero or negative, cannot find peaks.")
    else:
         print("No positive magnitude data to analyze.")
    # --- End Peak Finding ---

    plt.figure(figsize=(12, 6))

    plt.subplot(2,1,1)
    plt.plot(positive_frequencies, 20 * np.log10(positive_magnitude))  # Magnitude in dB
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Magnitude Spectrum (dB) - Frequencies")
    plt.grid(True)


    plt.subplot(2, 1, 2)
    plt.plot(positive_frequencies, positive_phase)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Spectrum - Frequencies')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_filter_results(error_signal, frequencies, psd, signal, sampling_rate, y_output,
                         filter_type="Filter", # Added argument, default "Filter"
                         threshold_ratio=0.1, prominence=None):
    """
    Plots the results of an adaptive filter, preserving the original 4-subplot structure.

    Args:
        error_signal (np.ndarray): The error signal e(n).
        frequencies (np.ndarray): Frequencies corresponding to the PSD array.
        psd (np.ndarray): Power Spectral Density estimate from filter weights.
        signal (np.ndarray): The original input signal d(n) (or desired signal).
        sampling_rate (float): The sampling rate of the signals.
        y_output (np.ndarray): The filter's output signal y(n).
        filter_type (str): Name of the filter used (e.g., "LMS", "NLMS", "RLS")
                           for plot titles and labels.
        threshold_ratio (float): Ratio of the maximum PSD height used as a minimum
                                 threshold for peak detection (0 to 1).
        prominence (float, optional): Required prominence of peaks for find_peaks.
                                      See scipy.signal.find_peaks documentation.
    """
    plt.figure(figsize=(12, 8)) # Keep original figure size

    # Plot Error Signal
    plt.subplot(2, 2, 1)
    # Assuming error_signal length corresponds to iterations
    plt.plot(error_signal)
    plt.xlabel("Iteration") # Keep original label
    plt.ylabel("Error")
    plt.grid(True)
    plt.title(f"{filter_type} Error Signal") # Use filter_type

    # --- Extract Positive Frequencies for Filter PSD ---
    # Keep original variable names but replace "lms" with "filter" for clarity
    positive_freq_indices_filter = np.where(frequencies >= 0)[0]
    if len(positive_freq_indices_filter) == 0:
        print(f"Warning: No positive frequencies found for {filter_type} spectrum analysis.")
        positive_frequencies_filter = np.array([])
        positive_psd_filter = np.array([])
    else:
        positive_frequencies_filter = frequencies[positive_freq_indices_filter]
         # Handle potential mismatch if psd passed in is already positive-only
        if len(psd) == len(frequencies):
             positive_psd_filter = psd[positive_freq_indices_filter]
        elif len(psd) == len(positive_freq_indices_filter):
             positive_psd_filter = psd # Assume psd was already processed
        else:
             print(f"Warning: Mismatch between frequencies ({len(frequencies)}) and psd ({len(psd)}) lengths for {filter_type}.")
             positive_psd_filter = np.array([]) # Cannot reliably extract

    # --- Peak Finding (using filter variables) ---
    print(f"\n--- {filter_type} Peak Frequencies ---") # Use filter_type
    peak_frequencies_filter = [] # Initialize
    peak_indices_filter = []
    # Check on the potentially extracted positive PSD
    valid_psd_for_peaks = positive_psd_filter[np.isfinite(positive_psd_filter)]

    if len(valid_psd_for_peaks) > 0 and np.max(valid_psd_for_peaks) > 1e-12:
        min_height_filter = np.max(valid_psd_for_peaks) * threshold_ratio
        try:
             peak_indices_filter, _ = find_peaks(positive_psd_filter, height=min_height_filter, prominence=prominence)
             # Ensure indices are valid for frequency array
             valid_peak_indices = [idx for idx in peak_indices_filter if idx < len(positive_frequencies_filter) and np.isfinite(positive_psd_filter[idx])]
             peak_frequencies_filter = positive_frequencies_filter[valid_peak_indices]
             peak_indices_filter = valid_peak_indices # Update to only valid ones

             if len(peak_frequencies_filter) > 0:
                peak_frequencies_filter.sort()
                print(f"Found {len(peak_frequencies_filter)} peaks at frequencies (Hz):")
                print([f"{freq:.2f}" for freq in peak_frequencies_filter])
             else:
                print(f"No significant {filter_type} peaks found above threshold ({threshold_ratio*100:.1f}% of max) or with specified prominence.")
                print(f"Max {filter_type} PSD value (finite): {np.max(valid_psd_for_peaks):.4g}")
        except Exception as e: # Catch potential errors during find_peaks
             print(f"Error during peak finding for {filter_type}: {e}")

    elif len(positive_psd_filter) > 0:
         print(f"{filter_type} PSD is likely all zero, negative, or non-finite. Cannot find peaks.")
    else:
         print(f"No valid positive {filter_type} PSD data to analyze.")
    # --- End Peak Finding ---


    # Plot Adaptive PSD (Positive Frequencies) - Linear Scale as original
    plt.subplot(2, 2, 2)
    if len(positive_frequencies_filter) > 0 and len(positive_psd_filter) == len(positive_frequencies_filter):
        plt.plot(positive_frequencies_filter, positive_psd_filter, label=f'{filter_type} PSD') # Use filter_type
        # Add markers for detected peaks
        if len(peak_frequencies_filter) > 0:
             # Get PSD values at the valid peak indices
             peak_psd_values = positive_psd_filter[peak_indices_filter]
             plt.plot(peak_frequencies_filter, peak_psd_values, "x", color='red', markersize=8, label='Detected Peaks')
        plt.legend()
    else:
         plt.plot([], []) # Plot empty if no data
         plt.legend([f'{filter_type} PSD (No data)'])

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title(f"Adaptive Spectrum ({filter_type})") # Use filter_type
    plt.grid(True)
    plt.xlim(0, sampling_rate / 2)  # Limit x-axis to Nyquist frequency


    # Plot the filter's output (predicted part of the signal)
    plt.subplot(2, 2, 3)
    # Assuming signal and y_output have same length corresponding to samples
    plt.plot(signal, label="Original signal")
    plt.plot(y_output, label=f"{filter_type} filter output/prediction") # Use filter_type
    plt.title(f"{filter_type} Predicted signal part") # Use filter_type
    plt.xlabel("Time (samples)") # Keep original label
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # --- Compare with FFT based spectrum (Log Scale) ---
    # Keep this subplot as it was
    plt.subplot(2, 2, 4)
    fft_output_comp = np.fft.fft(signal)
    frequencies_fft_comp = np.fft.fftfreq(len(signal), 1/sampling_rate)
    psd_fft_comp = np.abs(fft_output_comp)**2

    # Extract positive frequencies for FFT comparison
    positive_freq_indices_fft_comp = np.where(frequencies_fft_comp >= 0)[0]
    if len(positive_freq_indices_fft_comp) > 0:
        positive_frequencies_fft_comp = frequencies_fft_comp[positive_freq_indices_fft_comp]
        positive_psd_fft_comp = psd_fft_comp[positive_freq_indices_fft_comp]

        # Plot FFT PSD only if valid data exists
        valid_fft_psd = positive_psd_fft_comp > 1e-15 # Use small threshold for log plot
        if np.any(valid_fft_psd):
             plt.semilogy(positive_frequencies_fft_comp[valid_fft_psd],
                          positive_psd_fft_comp[valid_fft_psd],
                          label='FFT PSD', alpha=0.7)

    # Plot Filter PSD (from positive_psd_filter) only if valid data exists
    if len(positive_psd_filter) > 0:
        valid_filter_psd = positive_psd_filter > 1e-15 # Use small threshold for log plot
        if np.any(valid_filter_psd):
            # Make sure indices align if filtering happened
            freqs_to_plot = positive_frequencies_filter[valid_filter_psd]
            psd_to_plot = positive_psd_filter[valid_filter_psd]
            if len(freqs_to_plot) == len(psd_to_plot):
                 plt.semilogy(freqs_to_plot, psd_to_plot,
                             label=f'{filter_type} PSD', alpha=0.9) # Use filter_type
            else:
                 print(f"Warning: Mismatch plotting {filter_type} PSD on log scale.")


    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (Log Scale)") # Label clarifies log scale
    plt.title(f"Comparison FFT vs {filter_type} PSD (Log Scale)") # Use filter_type
    plt.xlim(0, sampling_rate / 2)
    # Simplified bottom limit for robustness
    current_ylim = plt.ylim()
    plt.ylim(bottom=max(np.finfo(float).eps, current_ylim[0]/10 if current_ylim[0] > 0 else 1e-9)) # Avoid zero, adjust based on data
    plt.grid(True, which='both') # Add grid for log scale
    plt.legend()

    plt.tight_layout()
    plt.show()



# --- Example Usage ---
if __name__ == "__main__":
    # signal config
    start_time = 0
    end_time = 0.1 # Make time shorter for clearer individual plots
    sampling_rate = 2000 # Increase SR for higher freqs

    sine_dict = {"type":"sine", "freq": 100, "amp": 0.5, "phase": 0}
    # AM with a simple envelope function for demonstration
    def simple_env(t): return 0.8 + 0.2 * np.sin(2 * np.pi * 20 * t)
    am_dict = {"type":"am", "freq": 300, "envelope": simple_env, "phase": np.pi/2}
    chirp_dict = {'type': 'chirp', 'start_freq': 200, 'end_freq': 400, 'amp': 0.6}
    noise_dict = {'type': 'noise', 'mean': 0.0, 'std_dev': 0.05}

    # List of components to use
    signals_list = [sine_dict, chirp_dict, am_dict, noise_dict]
    # signals_list = [sine_dict, noise_dict] # Test with fewer components

    # running funcs
    time_array = generate_time_vector(start_time, end_time, sampling_rate)

    # Generate the combined signal (optional, not needed for plotting components)
    # signal = sum_signal_components(time_array, signals_list )
    # plt.figure()
    # plt.plot(time_array, signal)
    # plt.title("Combined Signal")
    # plt.show()

    # --- Plot individual components ---
    plot_signal_components(time_array, signals_list)