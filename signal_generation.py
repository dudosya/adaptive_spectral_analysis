import numpy as np


def sine_wave_generator(start_time=0, end_time=5, sampling_rate=1000, frequency=1, phase=0):
    time = np.linspace(start_time,end_time, int( (end_time-start_time) * sampling_rate))
    signal = np.sin(2*np.pi*frequency*time+phase)
    return signal



def sine_additions(start_time = 0, end_time = 5, sampling_rate = 1000, amplitude_list = [1,2,3] ,frequency_list = [1,2,3], phase_list = [0,0,0]):
    signal = 0
    for index,freq in enumerate(frequency_list):
        sine_wave = amplitude_list[index] * sine_wave_generator(start_time=start_time, end_time=end_time, sampling_rate=sampling_rate ,frequency=freq, phase=phase_list[index])
        signal += sine_wave

    return signal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    frequencies = [1,2,3]
    signal = sine_additions(frequency_list=frequencies)


    plt.plot(signal)
    plt.grid(True)
    plt.show()
