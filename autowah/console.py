import pyaudio
import numpy as np
import time
from numpy_ringbuffer import RingBuffer


from .envelope_follower import EnvelopeFollower
import matplotlib.pyplot as plt

# https://www.geeksforgeeks.org/check-data-type-in-numpy/
# https://stackoverflow.com/questions/56147161/python-matplotlib-update-plot-in-the-background
# https://dsp.stackexchange.com/questions/72292/dynamic-filter-in-real-time-audio
# https://stackoverflow.com/questions/40483518/how-to-real-time-filter-with-scipy-and-lfilter
# Variable Fc filters: A simple approach to design of linear phase FIR filters with variable characteristics (P. Jarske, Y. Neuvo and S. K. Mitra,)
# https://arxiv.org/pdf/1804.02891.pdf
p = pyaudio.PyAudio()


def run():

    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024 * 4
    HISTORY_LENGTH = CHUNK * 20
    ENVELOPE_FOLLOWER_FC = 30
    envelope_follower = EnvelopeFollower(ENVELOPE_FOLLOWER_FC, RATE)

    scope = {
        "in": RingBuffer(capacity=HISTORY_LENGTH),
        "envelope": RingBuffer(capacity=HISTORY_LENGTH),
    }

    for sig in scope.values():
        sig.extend(np.zeros(sig.maxlen))

    def callback(in_data, frame_count, time_info, flag):
        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)
        # Process data here
        global state_vector

        envelope = envelope_follower.run(audio_data)

        scope["in"].extend(audio_data)
        scope["envelope"].extend(envelope)

        return audio_data, pyaudio.paContinue

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

    while stream.is_active():
        fig = plt.figure()
        ax1: plt.Axes = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)
        ax1.set_ylim((-1, 1))
        ax2.set_ylim((-1, 1))
        plt.ion()

        (line1,) = ax1.plot(np.array(scope["in"]))
        (line2,) = ax2.plot(np.array(scope["envelope"]))

        plt.show()

        # pyplot.plot()
        while True:
            line1.set_ydata(np.array(scope["in"]))
            line2.set_ydata(np.array(scope["envelope"]))
            fig.canvas.flush_events()
            time.sleep(0.01)

        time.sleep(20)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()

    p.terminate()
