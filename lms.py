import numpy as np



def nlms_filter(signal, filter_order, norm_step_size=0.1, delay=1, reg_factor=1e-6):
    """
    Applies the Normalized LMS (NLMS) algorithm with zero-padding.
    Handles filter startup to provide output/error corresponding to the original signal length.
    Assumes a prediction scenario: d(n) = original_signal[n].

    Args:
        signal (np.ndarray): The original input signal.
        filter_order (int): The order (M) of the FIR filter (number of weights).
        norm_step_size (float): Normalized step size (mu), typically 0 < mu < 2.
        delay (int): Prediction delay (k >= 1). Input vector uses samples up to n-delay.
        reg_factor (float): Small positive regularization constant (epsilon).

    Returns:
        weights (np.ndarray): The final adapted filter weights.
        errors (np.ndarray): Error signal e(n) = d(n) - y(n) over the original signal's duration.
        y_out (np.ndarray): Filter output signal y(n) over the original signal's duration.
    """
    N_orig = len(signal)
    M = filter_order

    # --- Input Validation ---
    if M <= 0:
        raise ValueError("Filter order must be positive.")
    if delay < 1:
        raise ValueError("Delay must be at least 1 for prediction setup.")
    if not (0 < norm_step_size < 2):
        print(f"Warning: norm_step_size (mu={norm_step_size}) is typically chosen between 0 and 2.")
    # We need at least M+delay-1 past samples for the first calculation.
    # The earliest relevant desired signal d(n) corresponds to original signal index 0.
    # The algorithm needs input up to index n-delay.
    # The first full input vector is needed when n = M+delay-1 (relative to padded signal).

    # --- Zero-Padding ---
    # Prepend zeros to provide history for the initial samples.
    # Need enough zeros so that when calculating output for the first original sample
    # (which is at index 'pad_length' in the padded signal), the required past samples exist.
    pad_length = M + delay - 1
    # Pad only at the beginning
    padded_signal = np.pad(signal, (pad_length, 0), 'constant', constant_values=(0,))
    N_padded = len(padded_signal)

    # --- Initialization ---
    weights = np.zeros(M)       # Initial weights
    errors_padded = np.zeros(N_padded) # Stores error for the padded signal duration
    y_out_padded = np.zeros(N_padded)  # Stores output for the padded signal duration
    # Optional: weights_history = np.zeros((N_padded, M))

    # --- NLMS Algorithm Loop (operates on padded signal) ---
    # Start index relative to the padded signal. This is the first index 'n'
    # where a full input vector x ending at index n-delay is available.
    start_index = M + delay - 1 # This is equal to pad_length

    for n in range(start_index, N_padded):
        # Extract input vector x(n) from the padded signal history
        # x(n) = [padded_signal[n-delay], ..., padded_signal[n-delay-M+1]]
        x_input_vector = padded_signal[n-delay-M+1 : n-delay+1][::-1]

        # Calculate filter output y(n) = w(n)^T * x(n)
        y_out_padded[n] = np.dot(weights, x_input_vector)

        # Desired signal d(n) is the sample from the *padded* signal at index n.
        # This corresponds to the original signal shifted by pad_length.
        d_n = padded_signal[n]

        # Calculate error e(n) = d(n) - y(n)
        errors_padded[n] = d_n - y_out_padded[n]

        # Calculate energy of the input vector ||x(n)||^2
        input_energy = np.dot(x_input_vector, x_input_vector)

        # Calculate adaptive step size
        adaptive_step = norm_step_size / (reg_factor + input_energy)

        # Update weights
        weights = weights + adaptive_step * errors_padded[n] * x_input_vector

        # Optional: Store weights
        # weights_history[n, :] = weights

    # --- Extract Results ---
    # Return the portions of error and output corresponding to the *original* signal's time frame.
    # These start at index pad_length in the padded arrays.
    errors = errors_padded[pad_length:]
    y_out = y_out_padded[pad_length:]

    # Optional: Trim weights history if needed
    # weights_history_trimmed = weights_history[pad_length:, :]

    # Ensure returned arrays match original signal length
    if len(errors) != N_orig or len(y_out) != N_orig:
         # This might happen if N_orig was very small, handle potential slicing issues
         # For simplicity here, we might return padded versions or raise an error
         # depending on desired behavior for edge cases. Let's trim/pad to ensure length.
         errors = errors[:N_orig] if len(errors) >= N_orig else np.pad(errors, (0, N_orig - len(errors)))
         y_out = y_out[:N_orig] if len(y_out) >= N_orig else np.pad(y_out, (0, N_orig - len(y_out)))
         print("Warning: Adjusting output array lengths due to potential edge case.")


    # return weights, errors, y_out, weights_history_trimmed # If history is needed
    return weights, errors, y_out




if __name__ == "__main__":
    from signal_generation import *
    from plot_funcs import *


    # signal config
    start_time = 0
    end_time = 0.5
    sampling_rate = 1000

    sine_dict = {"type":"sine", "freq": 100, "amp": 0.5, "phase": 0}
    am_dict = {"type":"am", "freq": 300, "envelope": None, "phase": 0}
    chirp_dict = {'type': 'chirp', 'start_freq': 200, 'end_freq': 210, 'amp': 1}

    #signals_list = [sine_dict]
    signals_list = [sine_dict,am_dict,chirp_dict]

    # running funcs
    time_array = generate_time_vector(start_time, end_time, sampling_rate)
    print(f"The signal has {len(time_array)} samples")


    signal = sum_signal_components(time_array, signals_list )
    plot_signal(signal, sampling_rate)



    # filter config
    filter_order = 20  
    norm_step_size = 0.1
    delay = 1 
    reg_factor=1e-6

    weights, error_signal, y_output = nlms_filter(signal, filter_order, norm_step_size, delay, reg_factor)
    frequencies, psd = adaptive_spectrum(weights, sampling_rate)
    plot_LMS_results(error_signal, frequencies, psd, signal, sampling_rate, y_output)