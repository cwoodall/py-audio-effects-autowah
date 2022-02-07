from scipy import signal
import numpy as np

@np.vectorize
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

    def __init__(self, filter_len=51, fs: float = None):
        """
        :param fs: Sample rate/frequency in Hz, if this is None then we assume 0-PI normalized inputs.
        :param filter_len:
        """

        self.fs = fs

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

        self.reset()

    def reset(self):
        self._compute_static_coefficients()
        self._d = 0
        self._is_init = False
        self._z = None

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

        # Normalize omega_c between 0 and pi (to obey nyquist)
        if self.fs:
            omega_c = 2 * np.pi * fc / self.fs
        else:
            omega_c = fc

        bs = [self._compute_coefficients(w) for w in omega_c]
        
        return np.array(
            [self._step(x,b) for x, b in zip(u, bs)],
            dtype=np.float32,
        )

    def _step(self, u, b):
        """
        Take one step forward in time with a given set of co-efficients for the filter.
        This handles keeping all of the filter state consistent between steps to make sure
        that the coefficients and parameters can change and still effect the output signal.

        :param u: a scalar value to step the filter forward with
        :param b: coefficients for the FIR filter
        :return   a filtered scalar value
        """
        if not self._is_init:
            self._z = signal.lfilter_zi(b, 1) * u
            self._is_init = True

        y, self._z = signal.lfilter(b, [1.0], [u], zi=self._z)
        return y[0]


    def _compute_coefficients(self, omega_c):
        return self.coefficients * _coefficient_sin(self.ns, omega_c)

    def _compute_static_coefficients(self):
        if self._is_odd:
            self.ns = np.array(range(-1 * self._N, self._N + 1))
            # This is the "Ideal LPF"
            h_M0 = lambda n: (self._w_co / np.pi) * np.sinc(self._w_co * n / np.pi)

            # Calculate a known set of coefficients for the FIR filter design
            self.coefficients = np.array(
                [
                    h_M0(n) * 1 / np.sin(self._w_co * n) if n != 0 else 1 / np.pi
                    for n in self.ns
                ]
            )
        else:
            raise NotImplementedError("Even case is not implemented")
