from scipy import signal
import numpy as np
import numba
from collections import deque

@numba.jit()
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

@numba.jit()
def _coefficient_sin(n, omega_c):
    if n != 0:
        return np.sin(omega_c * n)
    else:
        return omega_c

class VariableCutoffFilter:
    """
    Variable Cutoff Frequency Low Pass FIR filter.

    References:
        - [1] Petri Jarske, Yrjö Neuvo, Sanjit K. Mitra, A simple approach to the design of linear phase fir digital
          filters with variable characteristics, Signal Processing, Volume 14, Issue 4, 1988, Pages 313-326,
          ISSN 0165-1684, https://doi.org/10.1016/0165-1684(88)90090-4.
    """

    def __init__(self, filter_len=51, fs: float = None, chunk =None):
        """
        :param fs: Sample rate/frequency in Hz, if this is None then we assume 0-PI normalized inputs.
        :param filter_len:
        """

        self.fs = fs or 2*np.pi

        self._filter_len = filter_len

        # Calculate the N of positive non-zero terms in the filter. For odd the filter
        # length is 2*N + 1 (-N <= n <= N). The additional point is required to contain
        # the center point (n = 0).
        self._is_odd = filter_len % 2
        if self._is_odd:
            self._N = int((self._filter_len - 1) / 2)
            self._w_co = 2 * np.pi * (0.25 + 0.5 / (filter_len))
        else:
            # TODO: implement the even case
            raise NotImplementedError("Even case is not implemented")

        self.chunk = chunk
        if self.chunk:
            self.ys = np.empty(chunk, dtype=np.float32)

        self._coefficient_calc = np.vectorize(_coefficient_sin)

        self.reset()

    def reset(self):
        self._compute_static_coefficients()
        self._d = 0
        self._is_init = False
        self._z = deque(maxlen=self._filter_len)
        self._z.extend(np.zeros(self._filter_len, dtype=np.float32))

        # Create a lookup table for the available filter frequencies from 0 to nyquist
        self._coefficients_lut_size = 1024*2
        self._coefficients_lut_omegas = np.linspace(0, np.pi, endpoint=False, num=self._coefficients_lut_size)
        self._coefficients_lut = [self._compute_coefficients(w) for w in self._coefficients_lut_omegas]    
    
    def run(self, u, fc):
        """
        fc is converted to scale based on what fs is set to
        """
        # Convert u into an array if it is a scalar value
        if np.isscalar(u):
            u = np.array([u], dtype=np.float32)

        # Turn omega_c into an array
        if np.isscalar(fc):
            fc = np.array([fc] * len(u), dtype=np.float32)

        if not self.chunk:
            self.ys = np.empty(len(u), dtype=np.float32)

        # Normalize omega_c between 0 and pi (to obey nyquist)
        coeffs_idx = np.clip(np.round((self._coefficients_lut_size-1)*2*fc/self.fs),0,self._coefficients_lut_size-1)

        for i in range(len(u)):
            # coeffs_idx = find_nearest_idx(self._coefficients_lut_omegas, omega_c[i])
            coeffs = self._coefficients_lut[int(coeffs_idx[i])]
            self.ys[i] = self._step(u[i],coeffs)
        return self.ys

    def _step(self, u, b):
        """
        Take one step forward in time with a given set of co-efficients for the filter.
        This handles keeping all of the filter state consistent between steps to make sure
        that the coefficients and parameters can change and still effect the output signal.

        :param u: a scalar value to step the filter forward with
        :param b: coefficients for the FIR filter
        :return   a filtered scalar value
        """
        self._z.append(u)
        return sum([b[i] * self._z[i] for i in range(self._filter_len)])


    def _compute_coefficients(self, omega_c):
        return self.coefficients * self._coefficient_calc(self.ns, omega_c)

    def _compute_static_coefficients(self):
        if self._is_odd:
            self.ns = np.array(range(-1 * self._N, self._N + 1), dtype=np.float32)
            # This is the "Ideal LPF"... How can I increase the Q of this filter to get more resonance?
            h_M0 = lambda n: (self._w_co / np.pi) * np.sinc(self._w_co * n / np.pi)

            # Calculate a known set of coefficients for the FIR filter design
            self.coefficients = np.array(
                [
                    h_M0(n) * 1 / np.sin(self._w_co * n) if n != 0 else 1 / np.pi
                    for n in self.ns
                ], dtype=np.float32
            )
        else:
            raise NotImplementedError("Even case is not implemented")
