import numpy as np



def lms_filter(signal, filter_order, step_size, delay=1):
    """
    Applies the LMS algorithm to adapt an FIR filter.

    Args:
        signal: The input signal.
        filter_order: The order of the FIR filter (number of weights).
        step_size: The LMS step size (mu).
        delay: The delay (k) for the input to the filter.

    Returns:
        weights: The converged filter weights (NumPy array).
        errors: The error signal over time (for analysis).
    """
    N = len(signal)
    M = filter_order
    weights = np.zeros(M)  # Initialize weights to zero
    errors = np.zeros(N)
    y_out = np.zeros(N)

    for n in range(M + delay, N):  # Start after enough samples for delay
        x_delayed = signal[n-delay-M+1:n-delay+1][::-1] # Get delayed input, reversed for convolution
        y_out[n] = np.dot(weights, x_delayed)          # Filter output (prediction)
        errors[n] = signal[n] - y_out[n]                # Calculate error
        weights = weights + step_size * errors[n] * x_delayed  # Update weights

    return weights, errors, y_out

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
    W = np.fft.fft(weights)
    psd = np.abs(W)**2
    frequencies = np.fft.fftfreq(len(weights), 1/sampling_rate)
    return frequencies, psd
