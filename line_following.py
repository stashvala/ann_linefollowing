import pickle
import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from ev3dev.ev3 import *


class LineFollowing:

    MOTORS = [2, 3]
    COLOR_SENSORS = [4, 5]
    CONST_SPEED = 75

    def __init__(self, hidden_layer_sizes=(10, 5), solver="adam", lr=0.0001, epochs=1000, batch_size=32, alpha=0.001):
        self.MLP = MLPRegressor(hidden_layer_sizes, solver=solver, learning_rate_init=lr, alpha=alpha,
                                learning_rate='adaptive', shuffle=False, activation="relu", max_iter=epochs,
                                batch_size=batch_size, validation_fraction=0.1, early_stopping=True, verbose=True)

        # Ev3 inputs
        self.gs = None
        self.motor_left = None
        self.motor_right = None
        self.color_sensor_left = None
        self.color_sensor_right = None
        self.btn = None

    def setup_ev3(self):
        # GYRO SENSOR
        self.gs = GyroSensor()
        assert self.gs.connected, "Connect a gyro sensor to any sensor port"
        self.gs.mode = 'GYRO-ANG'

        # MOTORS
        self.motor_left = LargeMotor('outB')
        self.motor_right = LargeMotor('outD')

        # COLOR SENSOR
        self.color_sensor_left = ColorSensor('in1')
        assert self.color_sensor_left.connected, "Connect LEFT color sensor to sensor port 1"

        self.color_sensor_right = ColorSensor('in2')
        assert self.color_sensor_right.connected, "Connect RIGHT color sensor to sensor port 2"

        self.color_sensor_left.mode = 'COL-REFLECT'
        self.color_sensor_right.mode = 'COL-REFLECT'

        # BUTTON
        self.btn = Button()

    @staticmethod
    def motor_ratio(mot):
        return ((mot[:, 0] + 1) / (mot[:, 1] + 1) - 1) / 1000

    def train(self, train_csv, model_path="model/mlp.p"):

        data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

        # shift color inputs so that previous motor speed is input for current color
        x_motors = data[:-1, self.MOTORS]
        ratio = self.motor_ratio(x_motors)

        x_color = data[1:, self.COLOR_SENSORS] / 100  # normalize reflected color
        X = np.hstack((ratio[:, None], x_color))

        y_motors = data[1:, self.MOTORS]  # shift motor outputs
        y = self.motor_ratio(y_motors)

        self.MLP.fit(X, y)

        pickle.dump(self.MLP, open(model_path, "wb"))
        print("Model saved to ", model_path)

    def run(self, model_path=None):
        if model_path is not None:
            self.MLP = pickle.load(open(model_path, "rb"))
            print("Model {} loaded".format(model_path))
        else:
            print("Warning: using unloaded model!")

        self.setup_ev3()

        self.motor_left.run_forever(speed_sp=self.CONST_SPEED)
        self.motor_right.run_forever(speed_sp=self.CONST_SPEED)

        while not self.btn.any():  # While no button is pressed.
            left_color_intensity = self.color_sensor_left.value() / 100.
            right_color_intensity = self.color_sensor_right.value() / 100.

            # TODO: maybe remember prev predicted speed and use that
            curr_speed_left, curr_speed_right = self.motor_left.speed, self.motor_right.speed
            curr_speed_ratio = ((curr_speed_left + 1) / (curr_speed_right + 1) - 1) / 1000
            X = np.array([curr_speed_ratio, left_color_intensity, right_color_intensity])
            print("input: {:3.2f}\t{:1.3f}\t{:1.3f}".format(curr_speed_ratio,
                                                                     left_color_intensity, right_color_intensity))
            start_time = time.time()
            speed_ratio = self.MLP.predict(X.reshape(1, -1))[0]
            speed_right = self.CONST_SPEED * speed_ratio

            print("output: {:1.2f}, r_speed {:3.2f}, prediction took {:3.3f}ms".format(speed_ratio, speed_right, (time.time() - start_time) * 1000))

            self.motor_right.run_forever(speed_sp=speed_right)
            time.sleep(0.01)

        self.motor_left.stop(stop_action="hold")
        self.motor_right.stop(stop_action="hold")

if __name__ == '__main__':
    print("Program started")
    model = LineFollowing()
    # model.train("data/two_tracks.csv", "model/motor_ratio2.p")
    model.run("model/motor_ratio2.p")
