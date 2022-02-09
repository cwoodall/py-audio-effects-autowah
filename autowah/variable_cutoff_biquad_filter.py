from scipy import interpolate
import numpy as np
from collections import deque

 
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


        self.chunk = chunk
        if self.chunk:
            self.ys = np.empty(chunk, dtype=np.float32)

        self.b = np.zeros(3)
        self.a = np.zeros(2)

        K = np.vectorize(lambda x: np.tan(np.pi * x))
        points = np.linspace(0.001,.5,endpoint=True,num=64)
        self.K_interp = interpolate.interp1d(points, K(points))


        self.reset()

    # @numba.jit()
    def _calculate_lowpass_gains(self, wc, Q=.707):
        K = self.K_interp(wc)
        K_squared =  K * K
        norm = 1 / (1 + K / Q + K * K)
        
        self.b[0] = K_squared*norm
        self.b[1] = 2*self.b[0]
        self.b[2] = self.b[0]
        
        self.a[0] = 2*(K_squared-1) * norm
        self.a[1] = (1 - K/Q + K * K) * norm

    def reset(self):
        self._is_init = False
        self._z_b = deque(maxlen=2)
        self._z_b.extend(np.zeros(2))
        self._z_a = deque(maxlen=2)
        self._z_a.extend(np.zeros(2))
        # self._z.extend(np.zeros(self._filter_len, dtype=np.float32))

        # Qs = np.linspace(0,20,num=16)
        # fs = np.linspace(0,np.pi, num=1024)
        # points = np.meshgrid(Qs, fs)
        # self._gain_interpolator = interpolate.LinearNDInterpolator(points, _calculate_lowpass_gains)

        # # Create a lookup table for the available filter frequencies from 0 to nyquist
        # self._coefficients_lut_size = 1024*2
        # self._coefficients_lut_omegas = np.linspace(0, np.pi, endpoint=False, num=self._coefficients_lut_size)
        # self._coefficients_lut = [self._compute_coefficients(w) for w in self._coefficients_lut_omegas]    
    

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


        for i in range(len(u)):
            self._calculate_lowpass_gains(fc[i]/self.fs,2)
            y = self.b[0] * u[i] + self.b[1] * self._z_b[1] + self.b[2] * self._z_b[0] - self.a[0] * self._z_a[1] - self.a[1] * self._z_a[0]
            self._z_a.append(y)
            self._z_b.append(u[i])
            self.ys[i] = y
        return self.ys