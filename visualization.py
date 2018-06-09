import numpy as np
import matplotlib.pyplot as plt

MOTOR_COLS = [2, 3]
SENSOR_COLS = [4, 5]


def plot_motors_sensors(dataset, subsample=8):
    data = np.genfromtxt(dataset, delimiter=',', skip_header=1)

    track_ids = np.unique(data[:, 0])

    motor_color = 'b', 'cornflowerblue'
    sensor_color = 'g', 'mediumseagreen'

    for t_id in track_ids:
        rows = np.argwhere(data[:, 0] == t_id)[::subsample]
        motors = data[rows, MOTOR_COLS]
        sensors = data[rows, SENSOR_COLS]
        time = list(range(len(rows)))

        fig, ax1 = plt.subplots()

        ax1.plot(time, motors[:, 0], color=motor_color[0])
        ax1.plot(time, motors[:, 1], color=motor_color[1])

        ax1.set_ylabel('Hitrost motorja', color=motor_color[0])
        ax1.set_xlabel("Cas (ms)")
        ax1.tick_params('y', colors=motor_color[0])

        ax2 = ax1.twinx()

        ax2.plot(time, sensors[:, 0], color=sensor_color[0])
        ax2.plot(time, sensors[:, 1], color=sensor_color[1])

        ax2.set_ylabel('Intenziteta senzorja', color=sensor_color[0])
        ax2.tick_params('y', colors=sensor_color[0])

        plt.title("Track ID {:d}".format(int(t_id)))

        plt.show()

if __name__ == '__main__':
    plot_motors_sensors("data/two_tracks.csv")