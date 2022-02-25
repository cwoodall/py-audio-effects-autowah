import numpy as np
import numba


@numba.jit()
def _calculate_lowpass_gains(wc, Q):
    K = np.tan(np.pi * (wc))
    norm = 1 / (1 + K / Q + K * K)
    b0 = K * K * norm
    b1 = 2 * b0
    b2 = b0
    a1 = 2 * (K * K - 1) * norm
    a2 = (1 - K / Q + K * K) * norm
    return (b0, b1, b2, a1, a2)


@numba.jit()
def _calculate_bandpass_gains(wc, Q):
    K = np.tan(np.pi * (wc))
    norm = 1 / (1 + K / Q + K * K)
    b0 = K / Q * norm
    b1 = 0
    b2 = -b0
    a1 = 2 * (K * K - 1) * norm
    a2 = (1 - K / Q + K * K) * norm
    return (b0, b1, b2, a1, a2)


class VariableCutoffBiquadFilter:
    """
    Biquad Filter Implementation with Variable gain parameters. This assumes a LPF


    References:
        - [1]  https://www.earlevel.com/main/2011/01/02/biquad-formulas/
    """

    def __init__(self, fs: float = None, chunk=None, Q=2, filter_type="low"):
        """
        :param fs: Sample rate/frequency in Hz, if this is None then we assume 0-PI normalized inputs.
        """

        self.fs = fs or 2 * np.pi

        self.prev_u = np.zeros(2)

        self.Q = Q

        if filter_type not in ["bandpass", "low"]:
            raise Exception("Filter type must be low or bandpass")
        self.filter_type = filter_type

        self.chunk = chunk
        if self.chunk:
            self.ys = np.zeros(chunk, dtype=np.float32)
            self.dest_u = np.zeros(chunk + len(self.prev_u), dtype=np.float32)

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
            self.ys = np.zeros(len(u), dtype=np.float32)
            self.dest_u = np.zeros(len(u) + len(self.prev_u), dtype=np.float32)

        np.concatenate([u, self.prev_u], out=self.dest_u)

        for i in range(len(u)):
            # Calculate the minimal set of gains
            if self.filter_type == "low":
                b0, b1, b2, a1, a2 = _calculate_lowpass_gains(fc[i] / self.fs, self.Q)
            elif self.filter_type == "bandpass":
                b0, b1, b2, a1, a2 = _calculate_bandpass_gains(fc[i] / self.fs, self.Q)
            y = (
                b2 * self.dest_u[i - 2]
                + b1 * self.dest_u[i - 1]
                + b0 * self.dest_u[i]
                - a1 * self.ys[i - 1]
                - a2 * self.ys[i - 2]
            )
            self.ys[i] = y

        self.prev_u[0] = u[-2]
        self.prev_u[1] = u[-1]
        return self.ys
