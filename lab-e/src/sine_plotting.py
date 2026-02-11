import math

import numpy as np
import plotly.express as px


def make_sine_wave(A, f, start, stop):
    """
    Make a sine wave signal
    TO DO: replace range with a numpy array

    Returns: t: time samples
    v_out: voltage samples
    """
    t = range(start, stop)  # interpret as representing 1 s, 2 s, 3 s, ...
    v_out = [A * math.sin(2 * math.pi * f * time) for time in t]
    return t, v_out


def make_sine_wave_numpy(A, f, start, stop):
    """
    Make a sine wave signal
    Returns: t: time samples
    v_out: voltage samples
    """
    step = 1 / (f * 30)
    t = np.arange(
        start, stop, step
    )  # interpret as representing 1 s, 2 s, 3 s, ...
    v_out = A * np.sin(2 * np.pi * f * t)
    return t, v_out


def do_plots(t, v_out):
    fig = px.line(x=t, y=v_out, labels={'x': 'Time [s]', 'y': 'Voltage [V]'})
    fig.show()


if __name__ == "__main__":
    # Put plotting code below here
    