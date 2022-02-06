from scipy import signal
import numpy as np

class VariableCutoffFilter:
    """
    """

    def __init__(self, starting_bandwidth_Hz: float=100, sample_rate_Hz: float=44100, Q=.5, filter_len=51):
        """
        :param starting_bandwidth_Hz: Cutoff frequency to use in the lowpass filter stage
        :param sample_rate_Hz: Sample rate/frequency in Hz
        """

        self._filter_len = filter_len
        self._is_odd = filter_len % 2
        if self._is_odd:
            self._N = int((self._filter_len - 1) / 2)
            self._w_co = 2 * np.pi * (0.25 + .5 / (filter_len))
        else:
            # TODO: implement the even case
            raise Exception("Even case is not implemented")
            self._N = int((self._filter_len) / 2)
            self._w_co = 0.5 * np.pi

        self.reset()

    def reset(self):
        self._compute_static_coefficients()
        self._d = 0
        self._is_init = False
        self._z = None
  
    def run(self, u, omega_c):
        h0 = self._compute_coefficients(omega_c)

        if not self._is_init:
            self._z = signal.lfilter_zi(h0, 1) * u[0]
            self._is_init = True

        y, self._z = signal.lfilter(h0, [1.0], u, zi = self._z)
        return y

    def _compute_coefficients(self, omega_c):
        h0 = [c * np.sin(omega_c*n) if n != 0 else c*omega_c for c, n in zip(self.coefficients, self.ns)]
        return np.array(h0, dtype=np.float32)

    def _compute_static_coefficients(self):
        if self._is_odd:
            self.ns = range(-1*self._N, self._N+1)
            # This is the "Ideal LPF"
            h_M0 =lambda n: (self._w_co / np.pi) * np.sinc(self._w_co * n / np.pi)

            # Calculate a known set of coefficients for the FIR filter design
            self.coefficients = np.array([h_M0(n) * 1/np.sin(self._w_co * n) if n != 0 else 1/np.pi for n in self.ns])
        else:
            raise Exception("Even case is not implemented")

