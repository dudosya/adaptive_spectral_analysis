import numpy as np


def generate_time_vector(start_time=0, end_time=5, sampling_rate=1000):
    """Helper function to generate the time vector."""
    num_samples = int((end_time - start_time) * sampling_rate)
    time = np.linspace(start_time, end_time, num_samples, endpoint=False)
    return time


def sine_wave_generator(time, frequency=1, amplitude=1, phase=0):
    """Generates a sine wave on a given time vector."""
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return signal


def am_sine_generator(time, frequency=1, amplitude_envelope=None, phase=0):
    """
    Generates an Amplitude-Modulated (AM) sine wave.
    """
    if amplitude_envelope is None:
        amplitude = np.linspace(0.1, 1.0, len(time)) # Default envelope
    elif callable(amplitude_envelope):
        amplitude = amplitude_envelope(time)
    elif isinstance(amplitude_envelope, (np.ndarray, list)):
        if len(amplitude_envelope) != len(time):
            raise ValueError("Amplitude envelope array must have the same length as the time vector.")
        amplitude = np.asarray(amplitude_envelope)
    else:
         amplitude = amplitude_envelope # Assume scalar

    carrier_signal = np.sin(2 * np.pi * frequency * time + phase)
    modulated_signal = amplitude * carrier_signal
    return modulated_signal


def linear_chirp_generator(time, start_freq, end_freq, amplitude=1, phase=0):
    """
    Generates a linear chirp signal.
    """
    start_time = time[0]
    end_time = time[-1]
    duration = end_time - start_time

    if duration > 1e-9 : # Avoid division by zero
        k = (end_freq - start_freq) / duration
    else:
        k = 0

    time_relative = time - start_time
    instantaneous_phase = 2 * np.pi * (start_freq * time_relative + 0.5 * k * time_relative**2) + phase
    signal = amplitude * np.sin(instantaneous_phase)
    return signal


def gaussian_noise_generator(length, mean=0.0, std_dev=1.0):
    """
    Generates Gaussian (normal) distributed noise.

    Args:
        length (int): The number of noise samples to generate.
        mean (float): The mean (average) of the noise distribution. Defaults to 0.0.
        std_dev (float): The standard deviation of the noise distribution.
                         Controls the noise power/amplitude. Defaults to 1.0.

    Returns:
        np.ndarray: An array of noise samples of the specified length.
    """
    if std_dev < 0:
        raise ValueError("Standard deviation cannot be negative.")
    noise = np.random.normal(loc=mean, scale=std_dev, size=length)
    return noise


def sum_signal_components(time, component_list):
    """
    Sums multiple signal components defined on the same time vector,
    including optional Gaussian noise.

    Args:
        time (np.ndarray): The time vector.
        component_list (list): A list where each element is a dictionary
                               describing a signal component. Supported types:
                               - {'type': 'sine', 'freq': 2, 'amp': 0.5, 'phase': 0}
                               - {'type': 'am', 'freq': 5, 'envelope': func_or_array, 'phase': 0}
                               - {'type': 'chirp', 'start_freq': 1, 'end_freq': 10, 'amp': 1}
                               - {'type': 'noise', 'mean': 0.0, 'std_dev': 0.1}
    Returns:
        np.ndarray: The combined signal.
    """
    total_signal = np.zeros_like(time)
    signal_length = len(time) # Needed for noise generator

    for comp in component_list:
        comp_type = comp.get('type', 'sine').lower() # Default to sine

        if comp_type == 'sine':
             total_signal += sine_wave_generator(
                time,
                frequency=comp.get('freq', 1),
                amplitude=comp.get('amp', 1.0),
                phase=comp.get('phase', 0)
            )
        elif comp_type == 'am':
            total_signal += am_sine_generator(
                time,
                frequency=comp.get('freq', 1),
                amplitude_envelope=comp.get('envelope', 1.0), # Use a default scalar if None
                phase=comp.get('phase', 0)
            )
        elif comp_type == 'chirp':
             total_signal += linear_chirp_generator(
                time,
                start_freq=comp.get('start_freq', 1),
                end_freq=comp.get('end_freq', 1),
                amplitude=comp.get('amp', 1.0),
                phase=comp.get('phase', 0)
            )
        elif comp_type == 'noise':
             # Generate and add Gaussian noise
             mean = comp.get('mean', 0.0)
             std_dev = comp.get('std_dev', 0.1) # Default std dev if not specified
             total_signal += gaussian_noise_generator(
                 length=signal_length,
                 mean=mean,
                 std_dev=std_dev
             )
        else:
            print(f"Warning: Unknown component type '{comp_type}'. Skipping.")

    return total_signal


# --- Example Usage (within signal_generation.py) ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Configuration ---
    start_time = 0
    end_time = 0.05       # Shorter duration for clearer plot
    sampling_rate = 1000

    # --- Define Signal Components ---
    sine_dict = {"type": "sine", "freq": 5, "amp": 1.0, "phase": 0}

    # Example AM envelope: A slower sine wave controlling amplitude
    def am_envelope(t):
        return 0.5 + 0.4 * np.sin(2 * np.pi * 1 * t) # Varies between 0.1 and 0.9

    am_dict = {"type": "am", "freq": 20, "envelope": am_envelope, "phase": np.pi/4}
    chirp_dict = {'type': 'chirp', 'start_freq': 30, 'end_freq': 80, 'amp': 0.8}

    # --- Define Noise Component ---
    noise_dict = {'type': 'noise', 'mean': 0.0, 'std_dev': 1} # Adjust std_dev to control noise level

    # --- Combine Components ---
    # List includes deterministic signals and the noise specification
    signals_list = [sine_dict, am_dict, chirp_dict, noise_dict]
    # Example: Signal with only sine and noise
    # signals_list = [sine_dict, noise_dict]

    # --- Generate Time and Signal ---
    time = generate_time_vector(start_time, end_time, sampling_rate)
    total_signal = sum_signal_components(time, signals_list)

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.plot(time, total_signal, label='Combined Signal with Noise')

    # Optionally plot components without noise for comparison
    clean_signal_list = [comp for comp in signals_list if comp.get('type') != 'noise']
    clean_signal = sum_signal_components(time, clean_signal_list)
    plt.plot(time, clean_signal, label='Clean Signal (No Noise)', linestyle='--', alpha=0.7)

    plt.title('Generated Signal with Gaussian Noise')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()