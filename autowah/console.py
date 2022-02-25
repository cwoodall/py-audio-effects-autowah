from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple
import pyaudio
import numpy as np
import time
from autowah.control_values import ControlValueDef, ControlValues

from numpy_ringbuffer import RingBuffer

from multiprocessing import Process, Queue, Value, Lock

from .envelope_follower import EnvelopeFollower
from .variable_cutoff_filter import VariableCutoffFilter
from .variable_cutoff_biquad_filter import VariableCutoffBiquadFilter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

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


def plotter(scope: Dict, cv):
    scope_buffers = {
        k: RingBuffer(capacity=HISTORY_LENGTH) for k in scope.keys()
    }
    for sig in scope_buffers.values():
        sig.extend(np.zeros(sig.maxlen))

    plt.style.use("ggplot")

    fig = plt.figure()
    ax1: plt.Axes = fig.add_subplot(411)
    ax2 = fig.add_subplot(412, sharex=ax1)
    ax3 = fig.add_subplot(413, sharex=ax1)
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
    
    def update_value(cv_name: str, val):
        "test"
        cv.values[cv_name].value = val
        fig.canvas.draw_idle()
    
    def toggle_value(cv_name: str, __name):
        cv.values[cv_name].value = not cv.values[cv_name].value

    slider_axes = []
    sliders = []
    for i, x in enumerate(cv.values.values()):
        if x.typestr == 'b':
            # Make checkbuttons with all plotted lines with correct visibility
            rax = plt.axes([0.25, i*0.03+.01, 0.25, 0.03])
            # labels = [str(line.get_label()) for line in lines]
            # visibility = [line.get_visible() for line in lines]
            widget = CheckButtons(rax, [x.name], [x.value])
            widget.on_clicked(partial(toggle_value, x.name))
            sliders.append(widget)
        else:
            slider_axes.append(plt.axes([0.25, i*0.03+.01, 0.65, 0.03]))
            slider = Slider(
                ax=slider_axes[-1],
                label=x.name,
                valmin=x.min,
                valmax=x.max,
                valinit=x.value,
            )
            slider.on_changed(partial(update_value, x.name))
            sliders.append(slider)

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


def stream(scope, cv):
    p = pyaudio.PyAudio()

    # Make these variables controlable
    ENVELOPE_FOLLOWER_FC = 30
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    with cv.lock:
        lpf = VariableCutoffBiquadFilter(fs=RATE, chunk=CHUNK)
    

    def callback(in_data, frame_count, time_info, flag):
        lpf.Q = cv.values['Q'].value
        starting_freq = cv.values['starting_freq'].value
        sensitivity = cv.values['sensitivity'].value
        is_bandpass = cv.values['is_bandpass'].value
        if is_bandpass:
            lpf.filter_type = 'bandpass'
        else:
            lpf.filter_type = 'low'
        gain = cv.values['gain'].value
        mix = cv.values['mix'].value
        envelope_gain = cv.values['envelope_gain'].value

        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)

        # Process data here
        envelope = envelope_follower.run(audio_data)*envelope_gain
        freqs = starting_freq + envelope * sensitivity
        freqs = np.clip(freqs, .001, RATE/2-.1)

        out = mix*gain* lpf.run(audio_data, freqs) + audio_data*(1-mix)
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

    control_values = ControlValues(
        [
            ControlValueDef('is_bandpass', typestr = 'b', init_value = False),
            ControlValueDef('starting_freq', 'f', 100, 10, RATE/2 - .1),
            ControlValueDef('Q', 'f', 8.0, .1, 20),
            ControlValueDef('sensitivity', 'f', init_value = RATE/4, min=0, max=RATE/2),
            ControlValueDef('gain', 'f', init_value = .8),
            ControlValueDef('mix', 'f', init_value = .8),
            ControlValueDef('envelope_gain', 'f', init_value = 1, min=0, max=4),
        ])

    stream_proc = Process(target=stream, args=(scope,control_values,))
    plotter_proc = Process(target=plotter, args=(scope,control_values,))
    plotter_proc.start()
    stream_proc.start()


    stream_proc.join()
    plotter_proc.join()

