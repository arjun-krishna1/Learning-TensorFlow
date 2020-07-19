import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

"""
Some of this code is from the Coursera TensorFlow in Practice Specialization
The rest is adapted from their material
"""


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def naive_forecast(split_time, series):
    """
    PURPOSE: predict that the value of the next day is the same as the day before
    PARAMETERS:
        - series is a numpy array containing the time series data
        - split_time is the day after which the validation set begins
    OUTPUT:
        -a numpy array containing the predicted values. This array will start on the day after the start of series
        and will continue until the day after the end of the series
    """
    return series[split_time - 1:-1]


def moving_average_forecast(series, window_size):
    """
        PURPOSE: predict that the value of the next day is the mean of the last window_size days
        PARAMETERS:
            - series is a numpy array containing the time series data
            -  window_size is the number of days to average over
        OUTPUT:
            - a numpy array where output[i] is the mean of the last window_size days
    """
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)

# TODO: add differencing
