import numpy as np
import matplotlib.pyplot as plt

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



def plot_spectrum(signal, fft_output, sampling_rate = 1000):
    frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    magnitude = np.abs(fft_output)
    phase = np.angle(fft_output)
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_magnitude = magnitude[:len(magnitude)//2]
    positive_phase = phase[:len(phase)//2]


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


def plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output):
    plt.figure(figsize=(12, 6))

    # Plot Error Signal
    plt.subplot(2, 2, 1)
    plt.plot(error_signal)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.grid(True)
    plt.title("LMS Error Signal")

    # Plot Adaptive PSD (Positive Frequencies)
    plt.subplot(2, 2, 2)
    plt.plot(frequencies[:len(frequencies)//2], psd[:len(psd)//2])  # Only positive frequencies
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Adaptive Spectrum (LMS)")
    plt.grid(True)
    plt.xlim(0, sampling_rate / 2)  # Limit x-axis to Nyquist frequency

    # Plot the filter's output (predicted part of the signal)
    plt.subplot(2, 2, 3)
    plt.plot(signal, label="Original signal")
    plt.plot(y_output, label="LMS filter output/prediction")
    plt.title("LMS Predicted signal part")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Compare with FFT based spectrum in log scale (more informative)
    plt.subplot(2, 2, 4)
    fft_output = np.fft.fft(signal)
    frequencies_fft = np.fft.fftfreq(len(signal), 1/sampling_rate)
    psd_fft = np.abs(fft_output)**2
    plt.semilogy(frequencies_fft[:len(frequencies_fft)//2], psd_fft[:len(psd_fft)//2], label='FFT')
    plt.semilogy(frequencies[:len(frequencies)//2], psd[:len(psd)//2], label='LMS')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (dB)")
    plt.title("Comparison with FFT (Log Scale)")
    plt.xlim(0, sampling_rate / 2)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()  # Adjust subplot parameters for a tight layout
    plt.show()