from autowah.variable_cutoff_filter import VariableCutoffFilter
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import pytest

plt.style.use('ggplot')

def test_autowah_variable_cutoff_filter_even_lengths_not_supported():
    with pytest.raises(NotImplementedError):
         VariableCutoffFilter(filter_len=100)

def test_autowah_variable_cutoff_filter_coefficients():
    filter_len = 51
    vcf = VariableCutoffFilter(filter_len=filter_len)
    
    # Valid
    assert vcf._is_odd == True
    N = int((filter_len - 1)/2)
    assert vcf._N == N
    center_idx = N

    # Check that coefficients are symmetric as per the source paper [1]
    for n in range(1,N+1):
        assert vcf.coefficients[center_idx-n] == -1*vcf.coefficients[center_idx+n]

    # As defined by [1] the center value for an odd filter will always be 1/pi
    # This will be scaled appropriately
    assert vcf.coefficients[center_idx] == pytest.approx(1/np.pi)

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def complex_to_magnitude_dB(x):
    return 20*np.log10(np.abs(x))

def test_autowah_variable_cutoff_freq_response():
    filter_len = 51
    vcf = VariableCutoffFilter(filter_len=filter_len)
    N = int((filter_len - 1)/2)

    omega_cs = np.linspace(.1, np.pi, 10,endpoint=False)
    plt.tight_layout(pad=3.0)
    plt.subplot(211)
    plt.title(f'Variable Cutoff Frequency Filter response (len={filter_len})')
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Freq (rads)')
    plt.subplot(212)
    plt.title(f'Ideal H_ID(w) frequency response (len={filter_len})')
    plt.ylabel('Magnitude (dB)')
    plt.xlabel('Freq (rads)')

    for omega_c in omega_cs:
        w_uut, h_uut = signal.freqz(vcf._compute_coefficients(omega_c))
        plt.subplot(211)
        plt.plot(w_uut, complex_to_magnitude_dB(h_uut))

        h_uut_dB = complex_to_magnitude_dB(h_uut[find_nearest_idx(w_uut, omega_c)])
        # Check that the magnitude of the frequency response nearest the cutoff is ~-6 db. 
        assert pytest.approx(0.0, abs=.08) == np.abs((-6 - h_uut_dB)/6)

        # Ideal Sinc style filter with a sharp roll-off
        h_ID = [(omega_c / np.pi) * np.sinc(omega_c * n / np.pi) for n in range(-N, N+1)]
        w, h = signal.freqz(h_ID, [1])
        plt.subplot(212)

        plt.plot(w, complex_to_magnitude_dB(h))
        # Assert that the filter is within 1% of the ideal frequency response
        error = (np.abs((complex_to_magnitude_dB(h_uut)-complex_to_magnitude_dB(h))) / complex_to_magnitude_dB(h))
        assert pytest.approx(0, abs=.01) == error

    plt.suptitle("")
    plt.savefig('docs/test_results/test_autowah_variable_cutoff_freq_response.png')
    plt.close()

def test_autowah_variable_cutoff_time_response():
    RATE = 44100

    filter_len = 51
    vcf = VariableCutoffFilter(filter_len=filter_len)
    N = int((filter_len - 1)/2)

    freq = 10000
    # Calculate the normalized cutoff frequency
    omega_sin = (freq/RATE) * 2 * np.pi

    ts = np.arange(0, .1, 1/44100)
    input_us = [np.sin((10000*t)*(2*np.pi)) for t in ts]

    # Sweep the control frequency over almost the full range
    omegas = np.concatenate((
        np.linspace(0.1, np.pi,int(len(ts)/2)),
        np.linspace(np.pi, 0.1, int(len(ts)/2))))

    transition_band = (2000/RATE) * 2 * np.pi

    # Create an envelope to account for the bandwidth + the region that the transition is occuring over
    def envelope(omega, omega_c, transition_band):
        if omega < (omega_c-transition_band):
            return .1
        elif omega < (omega_c+transition_band):
            return 1.1
        else:
            return 1.05

    ys_max_values = [envelope(omega, omega_sin, transition_band) for omega in omegas]
    ys = vcf.run(input_us, omegas)

    plt.plot(ts, input_us)
    plt.plot(ts, ys_max_values)
    plt.plot(ts, ys)
    plt.savefig('docs/test_results/test_autowah_variable_cutoff_time_response_variable_fc.png')
    assert np.all(np.abs(ys) <= ys_max_values)

# [1]: [A simple approach to the design of linear phase FIR Digital Filters with Variable Characteristics]
