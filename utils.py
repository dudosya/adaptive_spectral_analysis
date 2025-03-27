import numpy as np


def adaptive_spectrum(weights, sampling_rate):
    """
    Calculates the power spectral density (PSD) from the filter weights.

    Args:
        weights: The converged filter weights.
        sampling_rate: The sampling rate of the original signal.

    Returns:
        frequencies: The frequencies corresponding to the PSD values.
        psd: The power spectral density.
    """
    M = len(weights)
    # Calculate the frequency response H(f) = FFT(w)
    W = np.fft.fft(weights)
    # Calculate PSD = |H(f)|^2 / M (optional normalization by M)
    # Often, for adaptive line enhancement/spectral analysis, 1/|A(f)|^2 is used
    # where A(f) is the frequency response of the *prediction error filter* (1, -w1, -w2, ...).
    # Let's compute the PSD of the estimated signal model, which is proportional to |W(f)|^2
    # Or, compute the inverse spectrum related to the prediction error filter A(z) = 1 - sum(w_k * z^-k)
    # Spectrum estimate S(f) = 1 / |A(exp(j*2*pi*f/fs))|^2
    # where A(exp(j*2*pi*f/fs)) = 1 - sum(w_k * exp(-j*2*pi*f*k/fs))

    # Create the prediction error filter coefficients [1, -w1, -w2, ..., -wM]
    a_coeffs = np.concatenate(([1], -weights))

    # Compute frequency response of the prediction error filter
    A_f = np.fft.fft(a_coeffs, n=max(M*4, 512)) # Use zero-padding for smoother spectrum

    # Compute PSD estimate = sigma_e^2 / |A(f)|^2 (sigma_e^2 is error variance, often assumed constant or estimated)
    # For simplicity, let's just plot 1 / |A(f)|^2 as the spectral shape indicator
    psd_estimate = 1.0 / (np.abs(A_f)**2 + 1e-10) # Add small epsilon for stability

    frequencies = np.fft.fftfreq(len(A_f), 1/sampling_rate)

    # Return only positive frequencies for easier plotting
    pos_freq_indices = frequencies >= 0
    
    # Normalize PSD for plotting (optional)
    psd_normalized = psd_estimate[pos_freq_indices]
    # psd_normalized = psd_normalized / np.max(psd_normalized) # Normalize to peak = 1
    
    return frequencies[pos_freq_indices], psd_normalized
