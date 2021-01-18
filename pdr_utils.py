import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import plotlyHelper
from scipy.signal import find_peaks
import itertools

sensor_types = ['Accelerometer', 'Gyroscope', 'Gravity', 'Barometer', 'Orientation']


class Step:
    def __init__(self, owner, length, mag, true_size=None):
        self.length = length
        self.magnitude = mag
        self.true_size = true_size
        self.approx_size = None
        self.owner = owner
        self.raw_signal = None
        self.Gk = None
        if self.true_size is not None:
            self.calc_gain()

    def calc_gain(self):
        self.Gk = self.true_size / (np.cbrt(self.magnitude / self.length))


def flatten_list(list_of_lists):
    return list(itertools.chain(*list_of_lists))


# ------------------------------------------------------ Loading utils ---------------------------------------------

def load_session(dirname, remove_ends_seconds=0, biases=False):
    sensor_data = {}
    fs = {}
    for sensor in sensor_types:
        try:
            filename = f'data/{dirname}/{sensor}.csv'
            df = pd.read_csv(filename, sep=',')
            df.sort_values(by='time', inplace=True)

            if sensor == 'Accelerometer':
                if biases is not False:
                    for axis, value in biases.items():
                        df[axis] = df[axis] - value[0]
                df['l2_norm'] = np.linalg.norm(df[['x', 'y', 'z']].values, axis=1)

            df['str_time'] = pd.to_datetime(df['time'], unit='ns')

            sensor_data[sensor] = df
            fs[sensor] = np.round(np.mean(1e9 / df['time'].diff()))

            if remove_ends_seconds:
                sensor_data[sensor] = sensor_data[sensor].iloc[
                                      round(remove_ends_seconds * fs[sensor]):-round(remove_ends_seconds * fs[sensor])]

        except IOError:
            print("No such sensor for this session")

    return sensor_data, fs


# ------------------------------------------------------ Plotting utils ---------------------------------------------

def plot_single_sensor(single_sensor_data, type, fig, col, row, fs=0, peaks=None, peakTH=2):
    for metric in single_sensor_data.columns[1:-1]:
        fig.add_trace(go.Scatter(x=list(single_sensor_data['str_time']),
                                 y=list(single_sensor_data[metric]),
                                 name=f'{type} ({metric})', mode='lines'),
                      col=col,
                      row=row)

    if peaks is not None:
        fig.add_trace(go.Scatter(x=list(single_sensor_data['str_time'][peaks]),
                                 y=list(single_sensor_data['l2_norm'][peaks]), mode='markers',
                                 marker=dict(size=8, color='red', symbol='cross'),
                                 name='Detected Peaks'),
                      col=col,
                      row=row)

        # fig.add_trace(go.Scatter(x=[single_sensor_data['str_time'].head(1), single_sensor_data['str_time'].tail(1)],
        fig.add_trace(
            go.Scatter(x=list(single_sensor_data['str_time'].head(1).append(single_sensor_data['str_time'].tail(1))),
                       y=[peakTH, peakTH],
                       mode='lines', line=dict(color="RoyalBlue", width=0.5), fillcolor="LightSkyBlue",
                       name='TH: ' + str(peakTH)),
            row=row, col=col)

    fig.layout.annotations[row - 1]['text'] = f'Plot of {type} sensor sampling @ {fs} Hz'

    fig.update_xaxes(title='time', tickfont_size=6, **plotlyHelper.axisStyle, row=row, col=1)
    fig.update_yaxes(title='Signal', tickfont_size=6, **plotlyHelper.axisStyle, row=row, col=1)


def plot_sensors(sensor_data, fs, peaks=None, title='', peakTH=2):
    fig = make_subplots(rows=len(sensor_data), cols=1, horizontal_spacing=0.055,
                        subplot_titles=['d' for _ in range(len(sensor_data))])

    row = 1
    for sensor, data in sensor_data.items():
        plot_single_sensor(data, sensor, fig, 1, row, fs=fs[sensor],
                           peaks=peaks if sensor == 'Accelerometer' else None,
                           peakTH=peakTH)
        row += 1

    for i in fig['layout']['annotations']:
        i['font']['size'] = 14
        i['xanchor'] = 'left'

    fig.update_layout(title_text=f'<b>Mobile phone sensor readings: {title}</b>', showlegend=True,
                      **plotlyHelper.layoutStyle)

    fig.show()


def plot_steps(list_of_steps):
    '''
    Plot all steps on-top of each other (just to get a sense of pattern_
    :param list_of_steps: list with Step objects
    :return: plots the steps aligned on top of each other
    '''
    fig = go.Figure()

    for step in list_of_steps:
        fig.add_trace(go.Scatter(x=np.arange(len(step.raw_signal)), y=step.raw_signal,
                                 mode='lines', line=dict(color="RoyalBlue", width=0.5), fillcolor="LightSkyBlue"))

    fig.show()


# -------------------------------------------------- Processing utils -------------------------------------------------

def calibrate(acc_vec, n_steps, distance, method='kim'):
    if method == 'kim':
        sk_true = distance / n_steps
        mean_acc = np.cbrt(acc_vec),
        gk = sk_true / mean_acc[0]
        return gk
    else:
        print('method not supported')
        return False


def collect_steps(data, fs, indices, owner, true_size=None, th=0.58):
    steps = []

    for i, indice in enumerate(indices[:-1]):
        indx_start = max(int(indice - 0.5 * th * fs), 0)
        indx_end = min(int(indice + 0.5 * th * fs), len(data['l2_norm']))

        abs(data['time'][indices[i + 1]] - data['time'][indices[i]])
        mag = sum(list(data['l2_norm'][indx_start:indx_end]))
        length = abs(indx_end - indx_start)
        steps.append(Step(owner=owner, length=length, mag=mag, true_size=true_size))
        steps[-1].raw_signal = list(data['l2_norm'][indx_start:indx_end])

    return steps


def calc_mean_gain(steps_list):
    flat_list = flatten_list(steps_list)
    return sum([step.Gk for step in flat_list]) / len(flat_list)


# ---------------------------------------------------- Testing utils --------------------------------------------------

def calc_per_step_errors(steps_list, gk):
    steps_list = flatten_list(steps_list)
    error = 0
    for step in steps_list:
        error += (gk * np.cbrt(step.magnitude / step.length) - step.true_size) ** 2
    return np.sqrt(error / len(steps_list))


def calc_per_walk_errors(walking_sessions_list, gk, true_distance=20):
    error = 0
    for walk in walking_sessions_list:
        error += ((true_distance-get_length(walk, gk))/true_distance)**2

    return np.sqrt(error / len(walking_sessions_list))


def get_length(steps_list, gk):
    length = 0
    for step in steps_list:
        length += gk * np.cbrt(step.magnitude / step.length)
    # print('Calculated Length =', length)
    return length


def test(acc_vec, n_steps, gk=None, method='kim'):
    if method == 'kim':
        if gk:
            return np.cbrt(acc_vec) * gk
        else:
            print('please input Gk gain param')
            return False
    else:
        print('method not supported')
        return False
