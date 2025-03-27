import numpy as np


def generate_time_vector(start_time=0, end_time=5, sampling_rate=1000):
    """Helper function to generate the time vector."""
    num_samples = int((end_time - start_time) * sampling_rate)
    # Use endpoint=False if you prefer excluding end_time, common in some signal processing
    time = np.linspace(start_time, end_time, num_samples, endpoint=False)
    return time


def sine_wave_generator(time, frequency=1, amplitude=1, phase=0):
    """Generates a sine wave on a given time vector."""
    # Amplitude is now an explicit parameter
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    return signal


def am_sine_generator(time, frequency=1, amplitude_envelope=None, phase=0):
    """
    Generates an Amplitude-Modulated (AM) sine wave.
    Non-stationary because the amplitude changes over time.
    """
    if amplitude_envelope is None:
        # Default to a simple linear ramp envelope if none provided
        amplitude = np.linspace(0.1, 1.0, len(time)) # Example envelope
    elif callable(amplitude_envelope):
        # If a function is provided, evaluate it at each time point
        amplitude = amplitude_envelope(time)
    elif isinstance(amplitude_envelope, (np.ndarray, list)):
        # If an array/list is provided, ensure it matches the time vector length
        if len(amplitude_envelope) != len(time):
            raise ValueError("Amplitude envelope array must have the same length as the time vector.")
        amplitude = np.asarray(amplitude_envelope)
    else:
         # Assume scalar amplitude (though that would be stationary)
         amplitude = amplitude_envelope

    # Generate the underlying sine wave with amplitude 1
    carrier_signal = np.sin(2 * np.pi * frequency * time + phase)
    # Modulate (multiply) by the time-varying amplitude
    modulated_signal = amplitude * carrier_signal
    return modulated_signal


def linear_chirp_generator(time, start_freq, end_freq, amplitude=1, phase=0):
    """
    Generates a linear chirp signal.
    Non-stationary because the instantaneous frequency changes linearly over time.
    """
    start_time = time[0]
    end_time = time[-1]
    duration = end_time - start_time

    # Calculate instantaneous frequency f(t) = start_freq + rate * t' where t' = time - start_time
    # Rate of frequency change (k)
    if duration > 1e-9 : # Avoid division by zero if duration is tiny
        k = (end_freq - start_freq) / duration
    else:
        k = 0

    # Phase is the integral of angular frequency (2*pi*f(t))
    # phi(t) = 2*pi * integral(start_freq + k*t') dt' + phase
    # phi(t) = 2*pi * (start_freq*t' + 0.5*k*t'^2) + phase
    time_relative = time - start_time
    instantaneous_phase = 2 * np.pi * (start_freq * time_relative + 0.5 * k * time_relative**2) + phase

    signal = amplitude * np.sin(instantaneous_phase)
    return signal


def sum_signal_components(time, component_list):
    """
    Sums multiple signal components defined on the same time vector.

    Args:
        time (np.ndarray): The time vector.
        component_list (list): A list where each element is a dictionary
                               describing a signal component, e.g.,
                               {'type': 'am', 'freq': 5, 'envelope': func, 'phase': 0}
                               {'type': 'chirp', 'start_freq': 1, 'end_freq': 10, 'amp': 1}
                               {'type': 'sine', 'freq': 2, 'amp': 0.5, 'phase': np.pi}
    Returns:
        np.ndarray: The combined signal.
    """
    total_signal = np.zeros_like(time)
    for comp in component_list:
        comp_type = comp.get('type', 'sine').lower() # Default to stationary sine
        if comp_type == 'am':
            total_signal += am_sine_generator(
                time,
                frequency=comp.get('freq', 1),
                amplitude_envelope=comp.get('envelope', 1.0),
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
        elif comp_type == 'sine': # Stationary sine
             total_signal += sine_wave_generator(
                time,
                frequency=comp.get('freq', 1),
                amplitude=comp.get('amp', 1.0),
                phase=comp.get('phase', 0)
            )
        else:
            print(f"Warning: Unknown component type '{comp_type}'. Skipping.")

    return total_signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    start_time = 0
    end_time = 5
    sampling_rate = 1000

    frequency = 1
    amplitude = 1
    phase = 0

    sine_dict = {"type":"sine", "freq": 2, "amp": 0.5, "phase": 0}
    am_dict = {"type":"am", "freq": 5, "envelope": None, "phase": 0}
    chirp_dict = {'type': 'chirp', 'start_freq': 1, 'end_freq': 10, 'amp': 1}

    signals_list = [sine_dict,am_dict,chirp_dict]


    time = generate_time_vector(start_time, end_time, sampling_rate)
    signal = sine_wave_generator(time, frequency, amplitude, phase)
    modulated_signal = am_sine_generator(time, frequency, None, phase)
    chirped_signal = linear_chirp_generator(time, 1,5, amplitude, phase)
    total_signal = sum_signal_components(time, signals_list )

    plt.plot(total_signal)
    plt.show()
