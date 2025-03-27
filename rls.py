import numpy as np

# --- RLS Algorithm Implementation ---
def rls_filter(signal, filter_order, forgetting_factor=0.99, delay=1, delta=1.0):
    """
    Applies the Recursive Least Squares (RLS) algorithm with zero-padding.
    Handles filter startup to provide output/error corresponding to the original signal length.
    Assumes a prediction scenario: d(n) = original_signal[n].

    Args:
        signal (np.ndarray): The original input signal.
        filter_order (int): The order (M) of the FIR filter (number of weights).
        forgetting_factor (float): Forgetting factor (lambda), typically 0 < lambda <= 1.
                                  Controls the memory of the algorithm. Values closer to 1
                                  provide longer memory.
        delay (int): Prediction delay (k >= 1). Input vector uses samples up to n-delay.
        delta (float): Small positive constant for initializing the inverse correlation matrix P.
                       Typically a small value (e.g., 0.1 to 10) or 1/epsilon where epsilon is small.
                       P(0) = delta * I.

    Returns:
        weights (np.ndarray): The final adapted filter weights.
        errors (np.ndarray): Error signal e(n) = d(n) - y(n) over the original signal's duration.
                              This is the a priori error.
        y_out (np.ndarray): Filter output signal y(n) over the original signal's duration.
                            Calculated using weights from the *previous* step (a priori).
    """
    N_orig = len(signal)
    M = filter_order

    # --- Input Validation ---
    if M <= 0:
        raise ValueError("Filter order must be positive.")
    if delay < 1:
        raise ValueError("Delay must be at least 1 for prediction setup.")
    if not (0 < forgetting_factor <= 1):
        raise ValueError("Forgetting factor (lambda) must be between 0 (exclusive) and 1 (inclusive).")
    if delta <= 0:
        raise ValueError("Initialization value (delta) must be positive.")
    if forgetting_factor < 0.9:
         print(f"Warning: Forgetting factor (lambda={forgetting_factor}) is quite low,")
         print("         which may lead to faster tracking but higher noise.")

    # --- Zero-Padding ---
    # Same padding logic as NLMS to handle initial conditions
    pad_length = M + delay - 1
    padded_signal = np.pad(signal, (pad_length, 0), 'constant', constant_values=(0,))
    N_padded = len(padded_signal)

    # --- Initialization ---
    weights = np.zeros(M)               # Initial weights w(0)
    P = delta * np.identity(M)          # Initial inverse correlation matrix P(0) = delta * I
    errors_padded = np.zeros(N_padded)  # Stores error for the padded signal duration
    y_out_padded = np.zeros(N_padded)   # Stores output for the padded signal duration
    # Optional: weights_history = np.zeros((N_padded, M))

    # --- RLS Algorithm Loop (operates on padded signal) ---
    # Start index relative to the padded signal.
    start_index = pad_length # M + delay - 1

    for n in range(start_index, N_padded):
        # Extract input vector x(n) from the padded signal history
        # x(n) = [padded_signal[n-delay], ..., padded_signal[n-delay-M+1]]
        x_input_vector = padded_signal[n-delay-M+1 : n-delay+1][::-1]

        # Calculate filter output y(n) = w(n-1)^T * x(n) (a priori estimate)
        # Note: 'weights' variable holds w(n-1) at the start of the iteration
        y_out_padded[n] = np.dot(weights, x_input_vector)

        # Desired signal d(n) is the sample from the *padded* signal at index n.
        d_n = padded_signal[n]

        # Calculate a priori error e(n) = d(n) - y(n)
        errors_padded[n] = d_n - y_out_padded[n]

        # --- RLS Update Steps ---
        # 1. Calculate intermediate vector pi(n) = P(n-1) * x(n)
        pi_vector = P @ x_input_vector # Using @ for matrix multiplication

        # 2. Calculate gain vector k(n) = pi(n) / (lambda + x(n)^T * pi(n))
        denominator = forgetting_factor + x_input_vector @ pi_vector
        gain_vector = pi_vector / denominator

        # 3. Update weights w(n) = w(n-1) + k(n) * e(n)
        weights = weights + gain_vector * errors_padded[n]

        # 4. Update inverse correlation matrix P(n) = (1/lambda) * [P(n-1) - k(n) * pi(n)^T]
        #    pi(n)^T = x(n)^T * P(n-1)
        P = (1 / forgetting_factor) * (P - np.outer(gain_vector, pi_vector))

        # Optional: Store weights
        # weights_history[n, :] = weights

    # --- Extract Results ---
    # Return the portions of error and output corresponding to the *original* signal's time frame.
    errors = errors_padded[pad_length:]
    y_out = y_out_padded[pad_length:]

    # Optional: Trim weights history if needed
    # weights_history_trimmed = weights_history[pad_length:, :]

    # Ensure returned arrays match original signal length
    if len(errors) != N_orig or len(y_out) != N_orig:
         # Same adjustment logic as in NLMS
         errors = errors[:N_orig] if len(errors) >= N_orig else np.pad(errors, (0, N_orig - len(errors)))
         y_out = y_out[:N_orig] if len(y_out) >= N_orig else np.pad(y_out, (0, N_orig - len(y_out)))
         print("Warning: Adjusting RLS output array lengths due to potential edge case.")

    # return weights, errors, y_out, weights_history_trimmed # If history is needed
    return weights, errors, y_out


# --- Adaptive Spectrum Function (Identical to yours) ---


# --- Main Execution Block ---
if __name__ == "__main__":
    from signal_generation import *
    from plot_funcs import *
    from utils import *

    # --- Signal Config ---
    start_time = 0
    end_time = 0.01 
    sampling_rate = 1000

    sine_dict = {"type": "sine", "freq": 100, "amp": 0.5, "phase": 0}
    #am_dict = {"type": "am", "freq": 300, "amp": 0.7, "mod_freq": 10, "mod_depth": 0.4, "phase": 0}
    am_dict = {'type': 'am', 'freq': 200, 'envelope': None, 'phase': 0}
    chirp_dict = {'type': 'chirp', 'start_freq': 400, 'end_freq': 450, 'amp': 0.6, 'phase': 0}

    signals_list = [sine_dict, am_dict, chirp_dict]
    # signals_list = [sine_dict, noise_dict] # Simpler signal

    # --- Generate Signal ---
    time_array = generate_time_vector(start_time, end_time, sampling_rate)
    print(f"The signal has {len(time_array)} samples")
    signal = sum_signal_components(time_array, signals_list)
    plot_signal(signal, sampling_rate) # Show input signal plot separately first

    # --- Filter Config ---
    filter_order = 30      # RLS can often use a slightly higher order effectively
    delay = 1



    # --- RLS ---
    print("\n--- Running RLS ---")
    forgetting_factor_rls = 0.995  # Typical value, closer to 1 means more memory
    delta_rls = 1.0             # Initialization for P matrix
    weights_rls, error_rls, y_rls = rls_filter(
        signal, filter_order, forgetting_factor_rls, delay, delta_rls
    )
    frequencies_rls, psd_rls = adaptive_spectrum(weights_rls, sampling_rate)
    plot_filter_results(error_rls, frequencies_rls, psd_rls, signal, sampling_rate, y_rls, "RLS")
    print(f"RLS Final Weights (first 5): {weights_rls[:5]}")
    print(f"RLS Final Error MSE: {np.mean(error_rls**2)}")