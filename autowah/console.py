import argparse
from email.mime import audio
from functools import partial
from multiprocessing.sharedctypes import Value
from typing import Dict
import pyaudio
import numpy as np
import time

import scipy.signal

from autowah.control_values import ControlValueDef, ControlValues
import wave
import click
import sys

from numpy_ringbuffer import RingBuffer
from multiprocessing import Process, Queue

from .envelope_follower import EnvelopeFollower
from .variable_cutoff_biquad_filter import VariableCutoffBiquadFilter
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

# https://www.geeksforgeeks.org/check-data-type-in-numpy/
# https://stackoverflow.com/questions/56147161/python-matplotlib-update-plot-in-the-background
# https://dsp.stackexchange.com/questions/72292/dynamic-filter-in-real-time-audio
# https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
# Variable Fc filters: A simple approach to design of linear phase FIR filters with variable characteristics (P. Jarske, Y. Neuvo and S. K. Mitra,)
# https://arxiv.org/pdf/1804.02891.pd

CHANNELS = 1
RATE = int(44100 / 4)
CHUNK = int(1024 / 2)
HISTORY_LENGTH = CHUNK * 20


def plotter(process_state: Value, scope: Dict, cv, args):
    scope_buffers = {k: RingBuffer(capacity=HISTORY_LENGTH) for k in scope.keys()}
    for sig in scope_buffers.values():
        sig.extend(np.zeros(sig.maxlen))

    plt.style.use("ggplot")

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax1: plt.Axes = fig.add_subplot(411)
    plt.title("Input Audio")
    ax2 = fig.add_subplot(412, sharex=ax1)
    plt.title("Envelope")
    ax3 = fig.add_subplot(413, sharex=ax1)
    plt.title("Output Audio")
    ax1.set_ylim((-1, 1))
    ax2.set_ylim((-1, 1))
    ax3.set_ylim((-1, 1))
    plt.ion()
    plt.suptitle(args.out)
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

    def toggle_value(cv_name: str):
        cv.values[cv_name].value = not cv.values[cv_name].value

    slider_axes = []
    sliders = []
    y_offset = 0.01
    for i, x in enumerate(cv.values.values()):
        if x.typestr == "b":
            # Make checkbuttons with all plotted lines with correct visibility
            rax = plt.axes([0.25, y_offset, 0.25, 0.1])
            y_offset += 0.1
            widget = CheckButtons(rax, [x.name], [x.value])
            widget.on_clicked(toggle_value)
            sliders.append(widget)
        else:
            slider_axes.append(plt.axes([0.25, y_offset, 0.65, 0.02]))
            y_offset += 0.02
            slider = Slider(
                ax=slider_axes[-1],
                label=x.name,
                valmin=x.min,
                valmax=x.max,
                valinit=x.value,
            )
            slider.on_changed(partial(update_value, x.name))
            sliders.append(slider)

    while process_state.value:
        for name, q in scope.items():
            while not q.empty():
                scope_buffers[name].extend(q.get_nowait())

        # Move this stuff into a different process...
        line1.set_ydata(np.array(scope_buffers["in"]))
        line2.set_ydata(np.array(scope_buffers["envelope"]))
        line3.set_ydata(np.array(scope_buffers["out"]))
        fig.canvas.flush_events()
        fig.canvas.draw()


def stream(process_state: Value, scope, cv, args):
    # open the file for reading.
    if args.input:
        wave_file = wave.open(args.input, 'rb')
    else:
        wave_file = None
    

    # Make these variables controlable
    ENVELOPE_FOLLOWER_FC = 20
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    # Anti aliasing filter
    anti_aliasing_filter_b, anti_aliasing_filter_a = scipy.signal.butter(10, RATE*.3, fs=wave_file.getframerate())
    anti_aliasing_filter_z = [scipy.signal.lfilter_zi(anti_aliasing_filter_b, anti_aliasing_filter_a).astype(np.float32)]

    with cv.lock:
        lpf = VariableCutoffBiquadFilter(fs=RATE, chunk=CHUNK)
    frames = []
    p = pyaudio.PyAudio()
    def callback(in_data, frame_count, time_info, flag):
        # Extract the control values
        lpf.Q = cv.values["Q"].value
        starting_freq = cv.values["starting_freq"].value
        sensitivity = cv.values["sensitivity"].value
        is_bandpass = cv.values["is_bandpass"].value
        if is_bandpass:
            lpf.filter_type = "bandpass"
        else:
            lpf.filter_type = "low"
        input_gain = cv.values["input_gain"].value
        gain = cv.values["gain"].value
        mix = cv.values["mix"].value
        envelope_gain = cv.values["envelope_gain"].value

        if wave_file:
            wave_freq = wave_file.getframerate()
            raw_data = wave_file.readframes(int(CHUNK*wave_freq/RATE))
            if raw_data:
                audio_data = np.fromstring(raw_data, dtype=np.int16)
                audio_data = audio_data.astype(np.float32, order='C') / 32768.0


                # Stereo to mono
                audio_data = (audio_data[0::2] + audio_data[1::2]) * .5
                audio_data, tmp = scipy.signal.lfilter(anti_aliasing_filter_b, anti_aliasing_filter_a, audio_data, zi=anti_aliasing_filter_z[0])
                anti_aliasing_filter_z[0] = tmp
                # Reinterpolate playback 
                audio_data = audio_data[0::int(wave_freq/RATE)]
            else: 
                stream.close()
                return None, pyaudio.paAbort

            if np.shape(audio_data)[0] != CHUNK:
                return None, pyaudio.paAbort
        else:
            # else:
            # using Numpy to convert to array for processing
            audio_data = np.fromstring(in_data, dtype=np.float32)


        audio_data = audio_data * input_gain
        # Apply the autowah effect
        # Calculate the envelope of the input waveform
        envelope = envelope_follower.run(audio_data) * envelope_gain

        # Calculate the cutoff frequencies and limit to less than nyquist to
        # prevent the variable parameter low pass filter from blowing up.
        fcs = starting_freq + envelope * sensitivity
        fcs = np.clip(fcs, 0.001, RATE / 2 - 0.1)

        # Apply the variable frequency low pass filter and also mix with the
        # original audio
        out = mix * gain * lpf.run(audio_data, fcs) + audio_data * (1 - mix)
        out = out.astype(np.float32)

        scope["in"].put_nowait(audio_data)
        scope["envelope"].put_nowait(envelope)
        scope["out"].put_nowait(out)

        frames.append(out)
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
    while stream.is_active() and process_state.value:
        time.sleep(1)

    wf = wave.open(args.out, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    frames = [np.clip(frame * 32768.0, -32768, 32767)  for frame in frames]
    frames = [frame.astype(np.int16) for frame in frames]
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    stream.close()
    p.terminate()
    process_state.value = False
    print("Closing audio stream")

def run():
    # Setup the scope queues for plotting
    scope = {
        "in": Queue(),
        "envelope": Queue(),
        "out": Queue(),
    }

    # Setup the control values for controllable variables.
    control_values = ControlValues(
        [
            ControlValueDef("is_bandpass", typestr="b", init_value=False),
            ControlValueDef("starting_freq", "f", 100, 10, RATE / 2 - 0.1),
            ControlValueDef("Q", "f", 8.0, 0.1, 20),
            ControlValueDef(
                "sensitivity", "f", init_value=RATE / 4, min=0, max=RATE / 2
            ),
            ControlValueDef("input_gain", "f", init_value=1.0, min = 0, max = 10.0),
            ControlValueDef("gain", "f", init_value=1.0, min = 0, max = 10.0),
            ControlValueDef("mix", "f", init_value=0.8),
            ControlValueDef("envelope_gain", "f", init_value=1, min=0, max=4),
        ]
    )

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-i", "--input", type=str, default="")
    parser.add_argument("-o", "--out", type=str, default="out.wav")

    # Add all control values to the command line arguments
    for cv in control_values.values.values():
        if cv.typestr == 'b':
            t = bool
        else:
            t = float
        parser.add_argument(f"--{cv.name}", type=t, default=cv.init_value)

    args = parser.parse_args()

    # Take the command line arguments and set the control values to their results
    for cv in control_values.values.values():
        cv.value = vars(args)[cv.name]

    process_state = Value('b')
    process_state.value = True
    processes = [
        Process(
            target=stream,
            args=(
                process_state,
                scope,
                control_values,
                args,
            ),
        ),
        Process(
            target=plotter,
            args=(
                process_state,
                scope,
                control_values,
                args,
            ),
        ),
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
