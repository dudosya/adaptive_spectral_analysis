import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks # <-- Import find_peaks

def plot_signal(signal, sampling_rate, title="Signal"):
    signal = np.asarray(signal)
    time = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)

    plt.figure()
    plt.plot(time, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()



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


def plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output, threshold_ratio=0.1, prominence=None):
    plt.figure(figsize=(12, 8))

    # Plot Error Signal
    plt.subplot(2, 2, 1)
    plt.plot(error_signal)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.title("LMS Error Signal")


    # --- Extract Positive Frequencies for LMS PSD ---
    positive_freq_indices_lms = np.where(frequencies >= 0)[0]
    if len(positive_freq_indices_lms) == 0:
        print("Warning: No positive frequencies found for LMS spectrum analysis.")
        # Still attempt to plot other subplots
        positive_frequencies_lms = np.array([])
        positive_psd_lms = np.array([])
    else:
        positive_frequencies_lms = frequencies[positive_freq_indices_lms]
        positive_psd_lms = psd[positive_freq_indices_lms]

    # --- Peak Finding (LMS) ---
    print("\n--- LMS Peak Frequencies ---")
    peak_frequencies_lms = [] # Initialize in case of issues
    peak_indices_lms = []
    if len(positive_psd_lms) > 0 and np.max(positive_psd_lms) > 0:
        min_height_lms = np.max(positive_psd_lms) * threshold_ratio
        peak_indices_lms, _ = find_peaks(positive_psd_lms, height=min_height_lms, prominence=prominence)
        peak_frequencies_lms = positive_frequencies_lms[peak_indices_lms]

        if len(peak_frequencies_lms) > 0:
            peak_frequencies_lms.sort()
            print(f"Found {len(peak_frequencies_lms)} peaks at frequencies (Hz):")
            print([f"{freq:.2f}" for freq in peak_frequencies_lms])
            # Optionally print heights:
            # print(f"Peak PSD values (linear): {[f'{positive_psd_lms[i]:.4f}' for i in peak_indices_lms]}")
        else:
            print(f"No significant LMS peaks found above threshold ({threshold_ratio*100:.1f}% of max) or with specified prominence.")
            print(f"Max LMS PSD value: {np.max(positive_psd_lms):.4f}")

    elif len(positive_psd_lms) > 0:
         print("LMS PSD is all zero or negative, cannot find peaks.")
    else:
         print("No positive LMS PSD data to analyze.")
    # --- End Peak Finding (LMS) ---


    # Plot Adaptive PSD (Positive Frequencies)
    plt.subplot(2, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], psd[:len(psd)//2])  # Only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Adaptive Spectrum (LMS)")
    plt.grid(True)
    plt.xlim(0, sampling_rate / 2)  # Limit x-axis to Nyquist frequency

    # Add markers for detected LMS peaks
    if len(peak_frequencies_lms) > 0:
        plt.plot(peak_frequencies_lms, positive_psd_lms[peak_indices_lms], "x", color='red', markersize=8, label='Detected Peaks')
        plt.legend()

    # Plot the filter's output (predicted part of the signal)
    plt.subplot(2, 2, 3)
    plt.plot(signal, label="Original signal")
    plt.plot(y_output, label="LMS filter output/prediction")
    plt.title("LMS Predicted signal part")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # --- Compare with FFT based spectrum (Log Scale) ---
    plt.subplot(2, 2, 4)
    # Calculate FFT spectrum here for comparison plot
    fft_output_comp = np.fft.fft(signal)
    frequencies_fft_comp = np.fft.fftfreq(len(signal), 1/sampling_rate)
    psd_fft_comp = np.abs(fft_output_comp)**2

    # Extract positive frequencies for FFT comparison
    positive_freq_indices_fft_comp = np.where(frequencies_fft_comp >= 0)[0]
    if len(positive_freq_indices_fft_comp) > 0:
        positive_frequencies_fft_comp = frequencies_fft_comp[positive_freq_indices_fft_comp]
        positive_psd_fft_comp = psd_fft_comp[positive_freq_indices_fft_comp]

        # Plot FFT PSD only if valid data exists
        valid_fft_psd = positive_psd_fft_comp > 0
        if np.any(valid_fft_psd):
             plt.semilogy(positive_frequencies_fft_comp[valid_fft_psd],
                          positive_psd_fft_comp[valid_fft_psd],
                          label='FFT PSD', alpha=0.7)

    # Plot LMS PSD only if valid data exists
    if len(positive_psd_lms) > 0:
        valid_lms_psd = positive_psd_lms > 0
        if np.any(valid_lms_psd):
            plt.semilogy(positive_frequencies_lms[valid_lms_psd],
                         positive_psd_lms[valid_lms_psd],
                         label='LMS PSD', alpha=0.9)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density") # Corrected label
    plt.title("Comparison FFT vs LMS PSD (Log Scale)")
    plt.xlim(0, sampling_rate / 2)
    plt.ylim(bottom=max(np.finfo(float).eps, np.min(positive_psd_lms[valid_lms_psd]) / 10) if np.any(valid_lms_psd) else 1e-6) # Adjust y-lim bottom
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()