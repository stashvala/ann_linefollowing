import pickle
import time

import numpy as np
from sklearn.neural_network import MLPRegressor

from sample_collector import SampleCollector
from robot import Robot


class LineFollowing:

    MOTORS = [2, 3]
    COLOR_SENSORS = [4, 5]
    CONST_SPEED = 75
    L_MOTOR_BASE = 291.0
    R_MOTOR_BASE = 291.0

    def __init__(self, hidden_layer_sizes=(10,), solver="adam", lr=0.0001, epochs=100, batch_size=32, alpha=0.01):
        self.MLP = MLPRegressor(hidden_layer_sizes, solver=solver, learning_rate_init=lr, alpha=alpha,
                                learning_rate='adaptive', shuffle=False, activation="relu", max_iter=epochs,
                                batch_size=batch_size, validation_fraction=0.1, early_stopping=True, verbose=True)

        self.robot = None

    @staticmethod
    def motor_ratio(mot):
        return ((mot[:, 0] + 1) / (mot[:, 1] + 1) - 1) / 1000

    def train(self, train_csv, model_path="model/mlp.p"):

        data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

        # shift color inputs so that previous motor speed is input for current color
        x_motors = data[:-1, self.MOTORS]
        x_color = data[1:, self.COLOR_SENSORS] / 100  # normalize reflected color
        X = np.hstack((x_motors, x_color))

        y_motors = data[1:, self.MOTORS]  # shift motor outputs
        y = y_motors  # shift motor outputs

        # mirror data
        X_mirror = X.copy()
        X_mirror[:, [0, 1]] = X[:, [1, 0]]
        X_mirror[:, [2, 3]] = X[:, [3, 2]]
        y_mirror = y.copy()
        y_mirror[:, [0, 1]] = y_mirror[:, [1, 0]]

        X = np.vstack((X, X_mirror))
        max_l_motor = np.max(X[:, 0])
        max_r_motor = np.max(X[:, 1])
        X[:, 0] /= max_l_motor
        X[:, 1] /= max_r_motor
        print("Max L motor = {}, Max R motor = {}".format(max_l_motor, max_r_motor))

        y = np.vstack((y, y_mirror))
        y[:, 0] /= max_l_motor
        y[:, 1] /= max_r_motor

        self.MLP.fit(X, y)

        pickle.dump(self.MLP, open(model_path, "wb"))
        print("Model saved to ", model_path)

    def init_robot(self):
        if self.robot is None:
            self.robot = Robot()

        if not self.robot.is_ev3_set():
            self.robot.setup_ev3()

    def run(self, model_path=None, out_log_file=None):
        if model_path is not None:
            self.MLP = pickle.load(open(model_path, "rb"))
            print("Model {} loaded".format(model_path))
        else:
            print("Warning: using unloaded model!")

        self.init_robot()

        out_log = []

        self.robot.drive(self.CONST_SPEED, self.CONST_SPEED)

        while not self.robot.btn.any():  # While no button is pressed.
            left_color_intensity = self.robot.color_sensor_left.value() / 100.
            right_color_intensity = self.robot.color_sensor_right.value() / 100.

            curr_speed_left, curr_speed_right = self.robot.motor_left.speed / self.L_MOTOR_BASE, \
                                                self.robot.motor_right.speed / self.R_MOTOR_BASE
            X = np.array([curr_speed_left, curr_speed_right, left_color_intensity, right_color_intensity])
            print("input: {:3.2f}\t{:3.2f}\t{:1.3f}\t{:1.3f}".format(curr_speed_left, curr_speed_right,
                                                                     left_color_intensity, right_color_intensity))
            start_time = time.time()
            speed_left, speed_right = self.MLP.predict(X.reshape(1, -1))[0]
            speed_left, speed_right = speed_left * self.L_MOTOR_BASE, speed_right * self.R_MOTOR_BASE

            print("output: {:3.2f}\t{:3.2f}, prediction took {:3.3f}ms".format(speed_left, speed_right,
                                                                               (time.time() - start_time) * 1000))

            self.robot.drive(speed_left, speed_right)

            if out_log_file is not None:
                out_log.append((speed_left, speed_right, left_color_intensity * 100, right_color_intensity * 100))

        self.robot.stop()

        print("Line following finished!")

        if out_log_file is not None:
            SampleCollector.create_csv(out_log_file, out_log)

if __name__ == '__main__':
    print("Program started")
    model = LineFollowing()

    model_path = "model/one_track_mirror.p"

    # model.train("data/one_track_better.csv", model_path)

    while True:
        model.run(model_path, "results/output.csv")
        run_again = input("Run again? [y/n]")
        if run_again != 'y':
            break
