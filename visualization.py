import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_motors_sensors_ann_out(out_file, subsample=1):
    data = np.genfromtxt(out_file, delimiter=',', skip_header=1)

    motor_color = 'b', 'cornflowerblue'
    sensor_color = 'g', 'mediumseagreen'

    motors = data[:, [0, 1]]
    sensors = data[:, [2, 3]]
    time = list(range(data.shape[0]))

    fig, ax1 = plt.subplots()

    ax1.plot(time, motors[:, 0], color=motor_color[0])
    ax1.plot(time, motors[:, 1], color=motor_color[1])

    ax1.set_ylabel('Hitrost motorja', color=motor_color[0])
    ax1.set_xlabel("Meritev")
    ax1.tick_params('y', colors=motor_color[0])

    ax2 = ax1.twinx()

    ax2.plot(time, sensors[:, 0], color=sensor_color[0])
    ax2.plot(time, sensors[:, 1], color=sensor_color[1])

    ax2.set_ylabel('Intenziteta senzorja', color=sensor_color[0])
    ax2.tick_params('y', colors=sensor_color[0])

    # plt.title("Ravna črta".format(int(t_id)))

    patches = [mpatches.Patch(color=motor_color[0], label='Levi motor'),
               mpatches.Patch(color=motor_color[1], label='Desni motor'),
               mpatches.Patch(color=sensor_color[0], label='Levi senzor'),
               mpatches.Patch(color=sensor_color[1], label='Desni senzor')]

    ax1.legend(handles=patches)

    plt.show()


def plot_motors_sensors_dataset(dataset, subsample=2):
    data = np.genfromtxt(dataset, delimiter=',', skip_header=1)

    track_ids = np.unique(data[:, 0])

    motor_color = 'b', 'cornflowerblue'
    sensor_color = 'g', 'mediumseagreen'

    for t_id in track_ids:
        rows = np.argwhere(data[:, 0] == t_id)[::subsample]
        motors = data[rows, [2, 3]]
        sensors = data[rows, [4, 5]]
        time = list(range(len(rows)))

        fig, ax1 = plt.subplots()

        ax1.plot(time, motors[:, 0], color=motor_color[0])
        ax1.plot(time, motors[:, 1], color=motor_color[1])

        ax1.set_ylabel('Hitrost motorja', color=motor_color[0])
        ax1.set_xlabel("Meritev")
        ax1.tick_params('y', colors=motor_color[0])

        ax2 = ax1.twinx()

        ax2.plot(time, sensors[:, 0], color=sensor_color[0])
        ax2.plot(time, sensors[:, 1], color=sensor_color[1])

        ax2.set_ylabel('Intenziteta senzorja', color=sensor_color[0])
        ax2.tick_params('y', colors=sensor_color[0])

        plt.title("Ravna črta".format(int(t_id)))

        plt.show()

if __name__ == '__main__':
    # plot_motors_sensors_dataset("data/straight_line.csv")

    plot_motors_sensors_ann_out("results/output_right.csv")
    plot_motors_sensors_ann_out("results/output_wrong.csv")
