from pdr_utils import *
import transforms3d
import plotlyHelper

'''
## General flow:

1. Load the data: Acc, Gyro (anything else?)

** apply calibration data

2. Set origin loc (P_0) and orientation init (W_0)
3. Generate orientation vector: 
    a. Rotate angular velocity
    b. Integrate to get next pose/attitude

4. Generate linear velocity vector:
    a. Rotate acceleration to inertial frame and integrate

5. Generate Position vector
    a. Integrate velocity

Gyro:Orientation:  
Y = roll
X = pitch
Z = yaw

'''


def raw2euler(angle_vec):
    TFM = np.array([[1, np.sin(angle_vec[0]) * np.tan(angle_vec[1]), np.cos(angle_vec[0]) * np.tan(angle_vec[1])],
                    [0, np.cos(angle_vec[0]), -np.sin(angle_vec[0])],
                    [0, np.sin(angle_vec[0]) / np.cos(angle_vec[1]), np.cos(angle_vec[0]) / np.cos(angle_vec[1])]])
    return TFM


def euler2tfm(euler_angle_vec):
    quaternion = transforms3d.euler.euler2quat(*euler_angle_vec)
    return transforms3d.quaternions.quat2mat(quaternion)


gyro_dict = {0: 'y', 1: 'x', 2: 'z'}


def get_calibration_values(calibration_dir, user='matan'):
    gyro_bias, acc_bias = {}, {}
    # acc_bias_full = np.zeros((3, 3))
    acc_scale = {}
    acc_bias_full = {'x': {}, 'y': {}, 'z': {}}
    axes = ['x', 'y', 'z']
    if user in ['matan', 'Matan']:
        suffix = '-M'
    else:
        suffix = ''

    for axis in axes:
        data, _ = load_session(f'{calibration_dir}/{axis}-calibration{suffix}')
        gyro = data['Gyroscope']
        acc = data['Accelerometer']
        gravity = data['Gravity']
        acc_with_gravity = acc
        acc_with_gravity = acc_with_gravity[['x', 'y', 'z']] + gravity[['x', 'y', 'z']]

        gyro_bias[axis], acc_bias[axis] = [gyro[axis].mean()], [acc[axis].mean()]

        acc_bias_full['x'][axis] = acc['x'].mean()
        acc_bias_full['y'][axis] = acc['y'].mean()
        acc_bias_full['z'][axis] = acc['z'].mean()

        acc_scale[axis] = [abs(acc_with_gravity[axis].mean()) / 9.8065]

        print(f'\nAcceleration Stats for axis {axis} pointing down')
        print(acc[['x', 'y', 'z']].describe())

    return pd.DataFrame.from_dict(gyro_bias), \
           pd.DataFrame.from_dict(acc_bias_full).max(axis=0), \
           pd.DataFrame.from_dict(acc_scale), \
           pd.DataFrame.from_dict(acc_bias_full)


def get_initial_alignment(gravity_vec):
    init_gravity = gravity_vec[['x', 'y', 'z']].head(5).mean()
    phi_0 = np.arctan2(-init_gravity['x'], -init_gravity['z'])
    theta_0 = np.arctan2(-init_gravity['y'], np.linalg.norm(init_gravity[['z', 'x']]))

    return phi_0, theta_0


# -------------------- Dead Reckoning ------------------------------------------------------------------------------

def dead_reckon(dir_name, remove_bias=False, title='', sma=0):

    pose, steps, _, name = dir_name.split('-')

    data, fs = load_session(dir_name)

    gyro = data['Gyroscope']
    dt_gyro = 1 / fs['Gyroscope']

    acc = data['Accelerometer']
    dt_acc = 1 / fs['Accelerometer']

    gravity = data['Gravity']
    phi_0, theta_0 = get_initial_alignment(gravity)

    if remove_bias:
        gyro_biases, acc_biases, acc_scale, acc_bias_per_axis = get_calibration_values('calibration', user=name)
        gyro[['x', 'y', 'z']] = gyro[['x', 'y', 'z']] - gyro_biases[['x', 'y', 'z']].to_numpy()
    else:
        acc_biases = pd.DataFrame({'x': [0], 'y': [0], 'z': [0]})
        acc_scale = pd.DataFrame({'x': [1], 'y': [1], 'z': [1]})
        acc_bias_per_axis = acc_biases
        gyro_biases = acc_biases

    if sma:  # simple-moving-average
        acc['x'] = acc['x'].rolling(window=sma, min_periods=1).mean()
        acc['y'] = acc['y'].rolling(window=sma, min_periods=1).mean()
        acc['z'] = acc['z'].rolling(window=sma, min_periods=1).mean()

    # -------------------- Init arrays: ------------------------------------------------------------------------------

    start_pose = np.array([phi_0, theta_0, 0])  # pose (attitude) at t=0
    start_loc = np.array([0, 0, 0])  # location at t=0
    start_velocity = np.array([0, 0, 0])  # velocity at t=0
    start_acc = np.array([0, 0, 0])  # acceleration (in ref frame) at t=0

    pose = np.array([start_pose])  # roll (phi), pitch (theta), yaw (psy)
    velocity = np.array([start_velocity])
    location = np.array([start_loc])
    acceleration = np.array([start_acc])

    # -------------------- Calc position and orientation: ------------------------------------------------------------

    for row_gyro, row_acc in zip(gyro.itertuples(), acc.itertuples()):
        angular_velocity = np.array([row_gyro.y-gyro_biases['y'], row_gyro.x-gyro_biases['x'], row_gyro.z-gyro_biases['z']])
        new_pose = pose[-1, :] + np.matmul(raw2euler(pose[-1, :]), angular_velocity) * dt_gyro
        pose = np.vstack((pose, new_pose))

        # in our case the gravity vector is already subtracted from the acceleration but we have other problems...
        tfm = euler2tfm(pose[-1, :])

        # as bias errors (in our sensor) are not constant per axis per orientation (weird non-linearity)
        # we calculate a mean bias term for each axis in the three main orientations (x,y,z)

        new_acc = np.matmul(tfm.T, np.array([(row_acc.y - acc_biases['y']) * acc_scale['y'],
                                           (row_acc.x - acc_biases['x']) * acc_scale['x'],
                                           (row_acc.z - acc_biases['z']) * acc_scale['z']])).squeeze()

        acceleration = np.vstack((acceleration, new_acc))

        new_v = velocity[-1, :] + np.mean([acceleration[-2, :], acceleration[-1, :]], axis=0) * dt_acc
        velocity = np.vstack((velocity, new_v))

        new_loc = location[-1, :] + np.mean([velocity[-1, :], velocity[-2, :]], axis=0) * dt_acc
        location = np.vstack((location, new_loc))

    # -------------------- Plot results: -----------------------------------------------------------------------------

    fig = go.Figure()

    for i, metric in enumerate(['roll', 'pitch', 'yaw']):
        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(acc[gyro_dict[i]]),
                                 name=f'Acc (raw) {gyro_dict[i]}', line=dict(width=4)))

        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(acceleration[:, i]),
                                 name=f'Acc (world frame) {gyro_dict[i]}', line=dict(width=4)))

        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(velocity[:, i]),
                                 name=f'Velocity {gyro_dict[i]}', line=dict(width=4, dash='dash')))

        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(location[:, i]),
                                 name=f'Position {gyro_dict[i]}', line=dict(width=4)))

        fig.add_trace(go.Scatter(x=list(acc['str_time']),
                                 y=list(pose[:, i]),
                                 name=metric, line=dict(width=4)))

    fig.update_xaxes(title='time', tickfont_size=6, **plotlyHelper.axisStyle)
    fig.update_yaxes(title='m', tickfont_size=6, **plotlyHelper.axisStyle)

    fig.update_layout(title_text=f'<b>{title}</b>', **plotlyHelper.layoutStyle)

    fig.show()

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter3d(x=list(location[:, 1]),
                                y=list(location[:, 0]),
                                z=list(location[:, 2]),
                                name=f'Position 3D plot', line=dict(width=4)))

    fig2.update_layout(title_text='3D position plot', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'))

    fig2.show()
