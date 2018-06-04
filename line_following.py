import pickle
from time import sleep

import numpy as np
from sklearn.neural_network import MLPRegressor
from ev3dev.ev3 import *


class LineFollowing:

    MOTORS = [2, 3]
    COLOR_SENSORS = [4, 5]

    def __init__(self, hidden_layer_sizes=(15, ), solver="adam", lr=0.00055, epochs=20000, batch_size=32):
        self.MLP = MLPRegressor(hidden_layer_sizes, solver=solver, learning_rate_init=lr,
                                learning_rate='adaptive', shuffle=False, activation="logistic", max_iter=epochs,
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

    def train(self, train_csv, model_path="model/mlp.p"):
        data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

        # shift color inputs so that previous motor speed is input for current color
        X = np.hstack((data[:-1, self.MOTORS], data[1:, self.COLOR_SENSORS]))
        X[:, [-2, -1]] /= 100  # normalize reflected color
        y = data[1:, self.MOTORS]  # shift motor outputs

        self.MLP.fit(X, y)

        pickle.dump(self.MLP, open(model_path, "wb"))
        print("model saved to ", model_path)

    def run(self, model_path=None):
        if model_path is not None:
            self.MLP = pickle.load(open(model_path, "rb"))
        else:
            print("Warning: using unloaded model!")

        self.setup_ev3()

        while not self.btn.any():  # While no button is pressed.
            left_color_intensity = self.color_sensor_left.value() / 100.
            right_color_intensity = self.color_sensor_right.value() / 100.

            # TODO: maybe remember prev predicted speed and use that
            curr_speed_left, curr_speed_right = self.motor_left.speed, self.motor_right.speed
            X = np.array([curr_speed_left, curr_speed_right, left_color_intensity, right_color_intensity])
            print("input: {:3.2f}\t{:3.2f}\t{:1.3f}\t{:1.3f}".format(curr_speed_left, curr_speed_right,
                                                                     left_color_intensity, right_color_intensity))

            speed_left, speed_right = self.MLP.predict(X)

            print("output: {:3.2f}\t{:3.2f}".format(speed_left, speed_right))

            self.motor_left.run_forever(speed_sp=speed_left)
            self.motor_right.run_forever(speed_sp=speed_right)
            sleep(0.01)

        self.motor_left.stop(stop_action="hold")
        self.motor_right.stop(stop_action="hold")

if __name__ == '__main__':
    model = LineFollowing()
    model.train("data/electrical_tape.csv", "model/validation.p")
    #model.run("model/mlp.p")
