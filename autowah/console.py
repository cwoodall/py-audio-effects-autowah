import pyaudio
import numpy as np
import time
from numpy_ringbuffer import RingBuffer
import wave
from multiprocessing import Process

from .envelope_follower import EnvelopeFollower
from .variable_cutoff_filter import VariableCutoffFilter
import matplotlib.pyplot as plt

# https://www.geeksforgeeks.org/check-data-type-in-numpy/
# https://stackoverflow.com/questions/56147161/python-matplotlib-update-plot-in-the-background
# https://dsp.stackexchange.com/questions/72292/dynamic-filter-in-real-time-audio
# https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
# Variable Fc filters: A simple approach to design of linear phase FIR filters with variable characteristics (P. Jarske, Y. Neuvo and S. K. Mitra,)
# https://arxiv.org/pdf/1804.02891.pd

def plotter():
    while True:
        time.sleep(1)

def run():
    p = pyaudio.PyAudio()

    CHANNELS = 1
    RATE = 44100
    CHUNK = int(1024 * 2)
    HISTORY_LENGTH = CHUNK * 20

    # Make these variables controlable
    ENVELOPE_FOLLOWER_FC = 50
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    # Q!
    lpf = VariableCutoffFilter(filter_len=41, fs=RATE, chunk=CHUNK)
    starting_freq = 50
    sensitivity = 20000
    scope = {
        "in": RingBuffer(capacity=HISTORY_LENGTH),
        "envelope": RingBuffer(capacity=HISTORY_LENGTH),
        "out": RingBuffer(capacity=HISTORY_LENGTH),
    }

    for sig in scope.values():
        sig.extend(np.zeros(sig.maxlen))

    def callback(in_data, frame_count, time_info, flag):
        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)
        # Process data here

        envelope = envelope_follower.run(audio_data)*10
        freqs = starting_freq + envelope * sensitivity

        # print(freqs)
        out = 1* lpf.run(audio_data, freqs)
        out = out.astype(np.float32)

        scope["in"].extend(audio_data)
        scope["envelope"].extend(envelope)
        scope["out"].extend(out)

        return out, pyaudio.paContinue


    # wf = wave.open('test.wav', 'rb')
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=CHANNELS,
        rate=RATE,
        frames_per_buffer=int(CHUNK),
        output=True,
        input=True,
        stream_callback=callback,
    )

    stream.start_stream()

    plt.style.use("ggplot")

    fig = plt.figure()
    ax1: plt.Axes = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax1.set_ylim((-1, 1))
    ax2.set_ylim((-1, 1))
    ax3.set_ylim((-1, 1))
    plt.ion()

    (line1,) = ax1.plot(np.array(scope["in"]))
    (line2,) = ax2.plot(np.array(scope["envelope"]))
    (line3,) = ax3.plot(np.array(scope["out"]))

    plt.show()

    # Move this stuff into a different process...
    while stream.is_active():
        line1.set_ydata(np.array(scope["in"]))
        line2.set_ydata(np.array(scope["envelope"]))
        line3.set_ydata(np.array(scope["out"]))
        fig.canvas.flush_events()
        # time.sleep(.1)
    #     time.sleep(20)
    #     stream.stop_stream()
    #     print("Stream is stopped")

    plotter_proc = Process(target=plotter)
    plotter_proc.start()
    plotter_proc.join()
    stream.close()

    p.terminate()
