from scipy import interpolate
import numpy as np
from collections import deque
import numba

K = np.vectorize(lambda x: np.tan(np.pi * x))
points = np.linspace(0,.5,endpoint=True,num=64)
K_interp = interpolate.interp1d(points, K(points))

@numba.jit()
def _calculate_lowpass_gains(wc, Q):
    K =  np.tan(np.pi * (wc))
    norm = 1 / (1 + K / Q + K * K)
    b0 = K*K*norm
    a0 = 2*(K*K-1) * norm
    a1 = (1 - K/Q + K * K) * norm
    return (b0, a0, a1)

@numba.jit()
def _calculate_bandpass_gains(wc, Q):
    K =  np.tan(np.pi * (wc))
    norm = 1 / (1 + K / Q + K * K)
    b0 = K / Q * norm;
    # b1 = 0;
    # b2 = -b0;
    a1 = 2 * (K * K - 1) * norm;
    a2 = (1 - K / Q + K * K) * norm;
    return (b0, a1, a2)

class VariableCutoffBiquadFilter:
    """
    Biquad Filter Implementation with Variable gain parameters. This assumes a LPF


    References:
        - [1]  https://www.earlevel.com/main/2011/01/02/biquad-formulas/
    """

    def __init__(self, fs: float = None, chunk=None):
        """
        :param fs: Sample rate/frequency in Hz, if this is None then we assume 0-PI normalized inputs.
        """

        self.fs = fs or 2*np.pi

        self.b = np.zeros(3)
        self.a = np.zeros(2)
        self.prev_u = np.zeros(2)

        self.chunk = chunk
        if self.chunk:
            self.ys = np.empty(chunk, dtype=np.float32)
            self.dest_u = np.empty(chunk + len(self.prev_u), dtype=np.float32)
     
        K = np.vectorize(lambda x: np.tan(np.pi * x))
        points = np.linspace(0.001,.5,endpoint=True,num=64)
        self.K_interp = interpolate.interp1d(points, K(points))

        self.reset()

    def reset(self):
        self._is_init = False


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
            self.dest_u = np.empty(len(u) + len(self.prev_u), dtype=np.float32)

        np.concatenate([u, self.prev_u], out=self.dest_u)

        for i in range(len(u)):
            # Calculate the minimal set of gains
            b0, a0, a1 = _calculate_lowpass_gains(fc[i]/self.fs,10)
            y = b0* self.dest_u[i-2] + b0*2 * self.dest_u[i-1] + b0 *self.dest_u[i] - a0 * self.ys[i-1] - a1 * self.ys[i-2]
            self.ys[i] = y

        self.prev_u[0] = u[-2]
        self.prev_u[1] = u[-1]
        return self.ys