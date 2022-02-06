from blinker import signal
from autowah import __version__
from autowah.variable_cutoff_filter import VariableCutoffFilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def test_autowah_version():
    assert __version__ == '0.1.0'

def test_autowah_variable_cutoff_filter_init():
    filter_len = 51
    vcf = VariableCutoffFilter(filter_len=filter_len)
    
    assert vcf._is_odd == True
    N = int((filter_len - 1)/2)
    assert vcf._N == N

    center_idx = N

    # Check that coefficients are symmetric as per the source paper [1]
    for n in range(1,N+1):
        assert vcf.coefficients[center_idx-n] == -1*vcf.coefficients[center_idx+n]
    assert vcf.coefficients[center_idx] == 1/np.pi

    ts = np.arange(0, 1, 1/44100)
    xs = [np.sin((10000*t)*(2*np.pi)) for t in ts]
    omegas = np.concatenate((
        np.linspace(0.1, np.pi,int(len(ts)/2)),
        np.linspace(np.pi, 0.1, int(len(ts)/2))))
    y = []
    for i,x in enumerate(xs):
        y.append(vcf.run([x],omegas[i]))
    plt.plot(ts, xs)
    plt.plot(ts, y)
    plt.show()

    omegas = np.linspace(.01, np.pi,10)
    for omega in omegas:
        w, h = signal.freqz(vcf._compute_coefficients(omega))
        plt.plot(w, 20 * np.log10(abs(h)))
        w, h = signal.freqz(*signal.butter(2, omega))
        plt.plot(w, 20 * np.log10(abs(h)))


    plt.show()
    assert False
# [1]: [A simple approach to the design of linear phase FIR Digital Filters with Variable Characteristics]
