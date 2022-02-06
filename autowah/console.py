from email.mime import audio
from numpy import linspace
import pyaudio
import numpy as np
import time
import scipy.signal

import queue
import matplotlib.pyplot as plt

# https://www.geeksforgeeks.org/check-data-type-in-numpy/
# https://stackoverflow.com/questions/56147161/python-matplotlib-update-plot-in-the-background
# https://dsp.stackexchange.com/questions/72292/dynamic-filter-in-real-time-audio
# https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
# Variable Fc filters: A simple approach to design of linear phase FIR filters with variable characteristics (P. Jarske, Y. Neuvo and S. K. Mitra,)
# https://arxiv.org/pdf/1804.02891.pdf
p = pyaudio.PyAudio()


class EnvelopeFollower:
    """
    """

    def __init__(self, bandwidth_Hz: float=100, sample_rate_Hz: float=44100):
        """
        :param bandwidth_Hz: Cutoff frequency to use in the lowpass filter stage
        :param sample_rate_Hz: Sample rate/frequency in Hz
        """

        # Create a lowpass filter with a 2nd order butterworth characteristic
        self._b, self._a = scipy.signal.butter(2, bandwidth_Hz, fs=sample_rate_Hz)

        # To use with pyaudio we need to retain the 32 bit float type to prevent unnecessary conversions
        self._b = self._b.astype(np.float32)
        self._a = self._a.astype(np.float32)

        # Store these parameters for getters later
        self._sample_rate_Hz = sample_rate_Hz
        self._bandwidth_Hz = bandwidth_Hz
        
        # Setup and then initialize the state vector
        self._z = None
        self.reset()

    def reset(self):
        """
        Reset the filter state
        """
        self._z = scipy.signal.lfilter_zi(self._b, self._a).astype(np.float32)

    def run(self, x):
        """
                # https://www.dsprelated.com/showarticle/938.php  Asynchronous Real Square-Law Envelope Detection

        """
        # Step 1: take the absolute value of the input signal
        abs_x = np.abs(x)

        # Step 2: apply a low pass filter to find the envelope of the signal
        y, self._z = scipy.signal.lfilter(self._b, self._a, abs_x, zi=self._z)
        return y

    @property
    def sample_rate_Hz(self):
        return self._sample_rate_Hz
    
    @property
    def bandwidth_Hz(self):
        return self._bandwidth_Hz

def run():

    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024*4

    ENVELOPE_FOLLOWER_FC = 30
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    global x1, x2, x1_prev, x2_prev
    x1 = np.array([0] * CHUNK)
    x2 = np.array([0] * CHUNK)
    x1_prev = np.array([0] * CHUNK)
    x2_prev = np.array([0] * CHUNK)

    def callback(in_data, frame_count, time_info, flag):
        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)
        # Process data here
        global state_vector

        out = envelope_follower.run(audio_data)
        
        global x1, x2, x1_prev, x2_prev
        x1_prev = x1
        x2_prev = x2
        x1 = out
        x2 = audio_data

        return audio_data, pyaudio.paContinue

    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=int(CHUNK),
                    output=True,
                    input=True,
                    stream_callback=callback)

    stream.start_stream()
    plt.style.use('ggplot')
    while stream.is_active():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylim((-1, 1))
        plt.ion()    
        line1, = ax.plot(np.append(x1_prev, x1))
        line2, = ax.plot(np.append(x2_prev, x2))

        plt.show()

        # pyplot.plot()
        while True:  
            line1.set_ydata(np.append(x1_prev, x1))
            line2.set_ydata(np.append(x2_prev, x2))
            fig.canvas.flush_events()
            time.sleep(.01)

        time.sleep(20)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()

    p.terminate()