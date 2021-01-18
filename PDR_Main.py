from pdr_utils import *
from navigation_utils import get_calibration_values
import os

DISTANCE = 20  # walking distance in meter


# ======================================== Calibrate Peak Detector ====================================================
def calibrate_peak_detector(resolution=30, peak_limits=(0.5, 4.5), distance_limits=(0.1, 1), bias=False):
    peak_to_test = np.round(np.linspace(peak_limits[0], peak_limits[1], resolution), 4)
    distances_to_test = np.round(np.linspace(distance_limits[0], distance_limits[1], resolution), 4)
    error_matrix = np.zeros((len(peak_to_test), len(distances_to_test), 1))

    for filename in os.listdir('data/walking'):
        temp_error_matrix = np.zeros((len(peak_to_test), len(distances_to_test)))
        pose, steps, _, name = filename.split('-')
        try:
            steps = int(steps)
        except Exception() as e:
            print('No valid number of steps in filename', e)

        data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)

        for i, pTH in enumerate(list(peak_to_test)):
            for j, dTH in enumerate(list(distances_to_test)):
                peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                                          distance=dTH * np.round(fs_calibrate['Accelerometer']),
                                          height=pTH)[0]

                steps_counted = len(peak_indices)
                temp_error_matrix[i, j] = (steps - steps_counted) / steps

        error_matrix = np.dstack((error_matrix, temp_error_matrix))

    error_matrix = np.sqrt(np.mean(error_matrix[:, :, 1:] ** 2, axis=2))
    min_indexes = np.where(error_matrix == np.amin(error_matrix))
    print(peak_to_test[min_indexes[0]], distances_to_test[min_indexes[1]])

    fig = go.Figure(data=go.Heatmap(
        x=distances_to_test,
        y=peak_to_test,
        z=error_matrix,
        type='heatmap'))

    fig.add_trace(go.Scatter(x=distances_to_test[min_indexes[1]],
                             y=peak_to_test[min_indexes[0]], mode='markers',
                             marker=dict(size=15, color='red', symbol='cross'),
                             name='minimum'))

    fig.update_yaxes(title='Peak TH value', tickfont_size=25, **plotlyHelper.axisStyle)
    fig.update_xaxes(title='Distance TH value', tickfont_size=25, **plotlyHelper.axisStyle)

    fig.update_layout(title_text=f'<b>Error vs Peak TH</b>', **plotlyHelper.layoutStyle)
    fig.show()

    return peak_to_test[min_indexes[0][0]], distances_to_test[min_indexes[1][0]]


# ======================================== Calculate Gk ==============================================================

def extract_all_steps(parent_dir, dTH=0.58, pTH=2, plot=False, owner='all', pose='all', bias=False):
    '''
    Extract the steps from
    :param dTH: TH for distance between peaks
    :param pTH: TH for minimal height of peaks
    :param plot: Boolean, set True to plot sensors
    :param owner: if set to specific name will only extract steps from specified owner
    :param pose: Only extract steps from a certain pose ('inear, inhand, pocket')
    :return: a list with Step objects extracted from all the files in the directory
    '''
    sensors_to_plot = ['Accelerometer', 'Barometer', 'Gravity', 'Gyroscope']
    steps_list = []
    for filename in os.listdir(parent_dir):

        pose, steps, _, name = filename.split('-')
        try:
            steps = int(steps)
        except Exception() as e:
            print('No valid number of steps in filename', e)

        if (owner == name or owner == 'all') and (pose == pose or pose == 'all'):
            data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)

            peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                                      distance=dTH * np.round(fs_calibrate['Accelerometer']),
                                      height=pTH)[0]

            if plot:
                plot_sensors({key: value for key, value in data_calibrate.items() if key in sensors_to_plot},
                             fs=fs_calibrate, peaks=peak_indices, title=filename)

            true_size = DISTANCE / steps
            steps_list.append(collect_steps(data_calibrate['Accelerometer'],
                                            fs_calibrate['Accelerometer'],
                                            peak_indices, name, true_size=true_size, th=dTH))

    print('Got all the steps!')
    return steps_list


def extract_steps_for_session(filename, dTH=0.58, pTH=2, plot=False, bias=False):
    sensors_to_plot = ['Accelerometer']
    pose, steps, _, name = filename.split('-')
    try:
        steps = int(steps)
    except Exception() as e:
        print('No valid number of steps in filename', e)

    data_calibrate, fs_calibrate = load_session(f'walking/{filename}', biases=bias)

    peak_indices = find_peaks(data_calibrate['Accelerometer']['l2_norm'],
                              distance=dTH * np.round(fs_calibrate['Accelerometer']),
                              height=pTH)[0]

    if plot:
        plot_sensors({key: value for key, value in data_calibrate.items() if key in sensors_to_plot},
                     fs=fs_calibrate, peaks=peak_indices, title=filename, peakTH=pTH)

    true_size = DISTANCE / steps
    steps_list = collect_steps(data_calibrate['Accelerometer'],
                               fs_calibrate['Accelerometer'],
                               peak_indices, name, true_size=true_size, th=dTH)

    return steps_list


# ======================================== Some tests ==============================================================

def test_cross_user_calibration(remove_bias=False, test_run=None):
    if remove_bias:
        _, biases, _, _ = get_calibration_values('calibration', user='Ido')
    else:
        biases = False

    pth, dth = calibrate_peak_detector(bias=biases)

    list_of_steps_ido = extract_all_steps(parent_dir='data/walking',
                                          owner='Ido',
                                          # pose='inear',
                                          bias=biases,
                                          dTH=dth,
                                          pTH=pth)

    list_of_steps_matan = extract_all_steps(parent_dir='data/walking',
                                            owner='Matan',
                                            # pose='inear',
                                            bias=biases,
                                            dTH=dth,
                                            pTH=pth)

    if remove_bias:
        print('Results with bias REMOVED:')
    else:
        print('Results with bias:')

    # ---- test calibrating User 1 and running on both:
    Gk = calc_mean_gain(list_of_steps_ido)
    print('Mean gain value (Gk) for Ido bias is:', Gk)

    step_error = round(calc_per_step_errors(list_of_steps_matan, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_matan, Gk), 3)
    print(f'RMS Error (Ido/Matan) is:{step_error} per step, {session_error} per walk')

    step_error = round(calc_per_step_errors(list_of_steps_ido, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_ido, Gk), 3)
    print(f'RMS Error (Ido/Ido) is:{step_error} per step, {session_error}m per walk')

    # ---- test calibrating on user 2 and running on both:
    Gk = calc_mean_gain(list_of_steps_matan)
    print('Mean gain value (Gk) bias for Matan is:', Gk)

    step_error = round(calc_per_step_errors(list_of_steps_ido, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_ido, Gk), 3)
    print(f'RMS Error (Matan/Ido) is:{step_error} per step, {session_error}m per walk')

    step_error = round(calc_per_step_errors(list_of_steps_matan, Gk), 3)
    session_error = round(calc_per_walk_errors(list_of_steps_matan, Gk), 3)
    print(f'RMS Error (Matan/Matan) is:{step_error} per step, {session_error}m per walk')

    # --- Plot a test file to see some graphs (along with peaks detected)
    if test_run is not None:
        try:
            extract_steps_for_session(test_run, plot=True, pTH=pth, dTH=dth, bias=biases)
        except Exception as e:
            print('No such test file found, exception:', e)


# ======================================== MAIN =====================================================================

filename = 'texting-27-steps-Matan'

test_cross_user_calibration(remove_bias=False, test_run=filename)
test_cross_user_calibration(remove_bias=True, test_run=filename)


# -- Other Tools and plots:  ----------------------------------------------------------------------------------------

# get_length(list_of_steps)

# plot_steps(list_of_steps)
