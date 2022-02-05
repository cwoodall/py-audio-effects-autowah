from email.mime import audio
from numpy import linspace
import pyaudio
import numpy as np
import time
import scipy.signal

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
    CHUNK = 1024*4

    b, a = scipy.signal.butter(2, 50, fs=RATE)
    b = b.astype(np.float32)
    a = a.astype(np.float32)

    # b = scipy.signal.firwin(150, 0.004)
    # a = 1
    global state_vector
    # Initialize the state
    state_vector = scipy.signal.lfilter_zi(b, a).astype(np.float32)

    global x1, x2, x1_prev, x2_prev
    x1 = np.array([0] * CHUNK)
    x2 = np.array([0] * CHUNK)
    x1_prev = np.array([0] * CHUNK)
    x2_prev = np.array([0] * CHUNK)

    def callback(in_data, frame_count, time_info, flag):
        # using Numpy to convert to array for processing
        audio_data = np.fromstring(in_data, dtype=np.float32)
        # Process data here
        global state_vector

        # square = audio_data*audio_data
        # fc = 2000*(np.sin(2*np.pi*time.time()*10) + 1)
        # print(fc)
        # b, a = scipy.signal.butter(3, fc, fs=RATE)
        # b = b.astype(np.float32)
        # a = a.astype(np.float32)

        
        # https://www.dsprelated.com/showarticle/938.php  Asynchronous Real Square-Law Envelope Detection
        envelope, state_vector = scipy.signal.lfilter(b, a, np.abs(audio_data), zi=state_vector)
        
        global x1, x2, x1_prev, x2_prev
        x1_prev = x1
        x2_prev = x2
        x1 = envelope
        x2 = audio_data
        # print(envelope.dtype)
        return audio_data, pyaudio.paContinue

    stream = p.open(format=pyaudio.paFloat32,
                    channels=CHANNELS,
                    rate=RATE,
                    frames_per_buffer=int(CHUNK),
                    output=True,
                    input=True,
                    stream_callback=callback)

    stream.start_stream()
    plt.style.use('ggplot')
    while stream.is_active():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ylim((-1, 1))
        plt.ion()    
        line1, = ax.plot(np.append(x1_prev, x1))
        line2, = ax.plot(np.append(x2_prev, x2))

        plt.show()

        # pyplot.plot()
        while True:  
            line1.set_ydata(np.append(x1_prev, x1))
            line2.set_ydata(np.append(x2_prev, x2))
            fig.canvas.flush_events()
            time.sleep(.01)

        time.sleep(20)
        stream.stop_stream()
        print("Stream is stopped")

    stream.close()

    p.terminate()