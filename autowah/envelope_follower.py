from scipy import signal
import numpy as np


class EnvelopeFollower:
    """ """

    def __init__(self, bandwidth_Hz: float = 100, sample_rate_Hz: float = 44100):
        """
        :param bandwidth_Hz: Cutoff frequency to use in the lowpass filter stage
        :param sample_rate_Hz: Sample rate/frequency in Hz
        """

        # Create a lowpass filter with a 2nd order butterworth characteristic
        self._b, self._a = signal.butter(2, bandwidth_Hz, fs=sample_rate_Hz)

        # To use with pyaudio we need to retain the 32 bit float type to prevent unnecessary conversions
        self._b = self._b.astype(np.float32)
        self._a = self._a.astype(np.float32)

        # Store these parameters for getters later
        self._sample_rate_Hz = sample_rate_Hz
        self._bandwidth_Hz = bandwidth_Hz

        # Setup and then initialize the state vector
        self._z = None
        self._is_init = False
        self.reset()

    def reset(self):
        """
        Reset the filter state
        """
        self._z = signal.lfilter_zi(self._b, self._a).astype(np.float32)
        self._is_init = False

    def run(self, x):
        """
        # https://www.dsprelated.com/showarticle/938.php  Asynchronous Real Square-Law Envelope Detection

        """
        if not self._is_init:
            self._is_init = True
            self._z = self._z * x[0]

        # Step 1: take the absolute value of the input signal
        abs_x = np.abs(x)


        # Step 2: apply a low pass filter to find the envelope of the signal
        y, self._z = signal.lfilter(self._b, self._a, abs_x, zi=self._z)
        return y

    @property
    def sample_rate_Hz(self):
        return self._sample_rate_Hz

    @property
    def bandwidth_Hz(self):
        return self._bandwidth_Hz
