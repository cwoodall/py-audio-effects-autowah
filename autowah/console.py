from typing import Dict
import pyaudio
import numpy as np
import time
from numpy_ringbuffer import RingBuffer

import wave
from multiprocessing import Process, Queue

from .envelope_follower import EnvelopeFollower
from .variable_cutoff_filter import VariableCutoffFilter
from .variable_cutoff_biquad_filter import VariableCutoffBiquadFilter
import matplotlib.pyplot as plt


# https://www.geeksforgeeks.org/check-data-type-in-numpy/
# https://stackoverflow.com/questions/56147161/python-matplotlib-update-plot-in-the-background
# https://dsp.stackexchange.com/questions/72292/dynamic-filter-in-real-time-audio
# https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
# Variable Fc filters: A simple approach to design of linear phase FIR filters with variable characteristics (P. Jarske, Y. Neuvo and S. K. Mitra,)
# https://arxiv.org/pdf/1804.02891.pd

CHANNELS = 1
RATE = int(44100/4)
CHUNK = int(1024/2) 
HISTORY_LENGTH = CHUNK * 20


def plotter(scope: Dict):
    scope_buffers = {
        k: RingBuffer(capacity=HISTORY_LENGTH) for k in scope.keys()
    }
    for sig in scope_buffers.values():
        sig.extend(np.zeros(sig.maxlen))

    plt.style.use("ggplot")

    fig = plt.figure()
    ax1: plt.Axes = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax1.set_ylim((-1, 1))
    ax2.set_ylim((-1, 1))
    ax3.set_ylim((-1, 1))
    plt.ion()

    (line1,) = ax1.plot(np.array(scope_buffers["in"]))
    (line2,) = ax2.plot(np.array(scope_buffers["envelope"]))
    (line3,) = ax3.plot(np.array(scope_buffers["out"]))
    plt.show()
    fig.canvas.draw()
    for name, q in scope.items():
        while not q.empty():
            scope_buffers[name].extend(q.get_nowait())

    while True:
        for name, q in scope.items():
            while not q.empty():
                scope_buffers[name].extend(q.get_nowait())
            
        # Move this stuff into a different process...
        line1.set_ydata(np.array(scope_buffers["in"]))
        line2.set_ydata(np.array(scope_buffers["envelope"]))
        line3.set_ydata(np.array(scope_buffers["out"]))
        fig.canvas.flush_events()
        fig.canvas.draw()

def stream(scope):
    p = pyaudio.PyAudio()

    # Make these variables controlable
    ENVELOPE_FOLLOWER_FC = 20
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    # Q!
    # if filter_type is "fir":
    #     lpf = VariableCutoffFilter(filter_len=31, fs=RATE, chunk=CHUNK)
    # else:
    lpf = VariableCutoffBiquadFilter(fs=RATE, chunk=CHUNK, Q=8)
    
    starting_freq = 10
    sensitivity = 10000

    def callback(in_data, frame_count, time_info, flag):
        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)
        # Process data here

        envelope = envelope_follower.run(audio_data)*1
        freqs = starting_freq + envelope * sensitivity

        out = .8* lpf.run(audio_data, freqs)
        out = out.astype(np.float32)

        scope["in"].put_nowait(audio_data)
        scope["envelope"].put_nowait(envelope)
        scope["out"].put_nowait(out)

        return out, pyaudio.paContinue

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
    while True:
        time.sleep(1)

    stream.close()

    p.terminate()

def run():
    scope = {
        "in": Queue(),
        "envelope": Queue(),
        "out": Queue(),
    }
    stream_proc = Process(target=stream, args=(scope,))
    plotter_proc = Process(target=plotter, args=(scope,))
    plotter_proc.start()
    stream_proc.start()


    stream_proc.join()
    plotter_proc.join()

