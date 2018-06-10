import pickle
import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from ev3dev.ev3 import *


class LineFollowing:

    MOTORS = [2, 3]
    COLOR_SENSORS = [4, 5]
    CONST_SPEED = 75
    L_MOTOR_BASE = 540.0
    R_MOTOR_BASE = 596.0

    L_SENSOR_WHITE_CALIB = -1
    R_SENSOR_WHITE_CALIB = -1
    L_SENSOR_BLACK_CALIB = -1
    R_SENSOR_BLACK_CALIB = -1

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
        print("Setting up ev3")

        # GYRO SENSOR
        self.gs = GyroSensor()
        assert self.gs.connected, "Connect a gyro sensor to any sensor port"
        self.gs.mode = 'GYRO-ANG'

        # MOTORS
        self.motor_left = LargeMotor('outB')
        self.motor_right = LargeMotor('outC')

        # COLOR SENSOR
        self.color_sensor_left = ColorSensor('in1')
        assert self.color_sensor_left.connected, "Connect LEFT color sensor to sensor port 1"

        self.color_sensor_right = ColorSensor('in2')
        assert self.color_sensor_right.connected, "Connect RIGHT color sensor to sensor port 2"

        self.color_sensor_left.mode = 'COL-REFLECT'
        self.color_sensor_right.mode = 'COL-REFLECT'

        # BUTTON
        self.btn = Button()

    def is_ev3_set(self):
        return self.motor_right is not None and self.motor_left is not None and \
               self.color_sensor_right is not None and self.color_sensor_right is not None

    def calibration_color_sensor_values(self, win_size=10):
        if self.L_SENSOR_WHITE_CALIB > 0 and self.R_SENSOR_WHITE_CALIB > 0 and \
           self.L_SENSOR_BLACK_CALIB > 0 and self.R_SENSOR_BLACK_CALIB > 0:

            print("Returning cached calibration values")
            return (self.L_SENSOR_WHITE_CALIB, self.R_SENSOR_WHITE_CALIB), \
                   (self.L_SENSOR_BLACK_CALIB, self.R_SENSOR_BLACK_CALIB)

        if not self.is_ev3_set:
            self.setup_ev3()

        print("Color sensor calibration started...\n")
        print("Press any key on robot and start driving it on WHITE surface for a bit")
        while not self.btn.any():
            time.sleep(0.01)

        Sound.beep().wait()
        time.sleep(0.5)
        l_sensor_white_avg = np.mean([self.color_sensor_left.value() for _ in range(win_size)])
        r_sensor_white_avg = np.mean([self.color_sensor_left.value() for _ in range(win_size)])
        Sound.beep().wait()

        print("Sensor white color avg: left = {:2.2f}, right = {:2.2f}".format(l_sensor_white_avg, r_sensor_white_avg))

        print("Press any key on robot and start driving it on BLACK surface for a bit")
        while not self.btn.any():
            time.sleep(0.01)
        Sound.beep().wait()

        Sound.beep().wait()
        time.sleep(0.5)
        l_sensor_black_avg = np.mean([self.color_sensor_left.value() for _ in range(win_size)])
        r_sensor_black_avg = np.mean([self.color_sensor_left.value() for _ in range(win_size)])
        Sound.beep().wait()

        print("Sensor black color avg: left = {:2.2f}, right = {:2.2f}".format(l_sensor_black_avg, r_sensor_black_avg))

        self.L_SENSOR_WHITE_CALIB, self.R_SENSOR_WHITE_CALIB = l_sensor_white_avg, r_sensor_white_avg
        self.L_SENSOR_BLACK_CALIB , self.R_SENSOR_BLACK_CALIB = l_sensor_black_avg, r_sensor_black_avg

        return (self.L_SENSOR_WHITE_CALIB, self.R_SENSOR_WHITE_CALIB), \
               (self.L_SENSOR_BLACK_CALIB, self.R_SENSOR_BLACK_CALIB)

    def calibrate_color_sensors(self, color_sensor_vals):
        left_sensor = np.interp(color_sensor_vals, [1, 100], [self.L_SENSOR_BLACK_CALIB)
        right_sensor

        return np.hstack((left_sensor, right_sensor))

    @staticmethod
    def motor_ratio(mot):
        return ((mot[:, 0] + 1) / (mot[:, 1] + 1) - 1) / 1000

    def train(self, train_csv, model_path="model/mlp.p"):

        data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

        # shift color inputs so that previous motor speed is input for current color
        x_motors = data[:-1, self.MOTORS]
        max_l_motor = np.max(x_motors[:, 0])
        max_r_motor = np.max(x_motors[:, 1])
        x_motors[:, 0] /= max_l_motor
        x_motors[:, 1] /= max_r_motor
        print("Max L motor = {}, Max R motor = {}".format(max_l_motor, max_r_motor))

        x_color = data[1:, self.COLOR_SENSORS] / 100  # normalize reflected color
        X = np.hstack((x_motors, x_color))

        y_motors = data[1:, self.MOTORS]  # shift motor outputs
        y_motors[:, 0] /= max_l_motor
        y_motors[:, 1] /= max_r_motor
        y = y_motors  # shift motor outputs

        self.MLP.fit(X, y)

        pickle.dump(self.MLP, open(model_path, "wb"))
        print("Model saved to ", model_path)

    def run(self, model_path=None):
        if model_path is not None:
            self.MLP = pickle.load(open(model_path, "rb"))
            print("Model {} loaded".format(model_path))
        else:
            print("Warning: using unloaded model!")

        if not self.is_ev3_set:
            self.setup_ev3()

        self.motor_left.run_forever(speed_sp=self.CONST_SPEED)
        self.motor_right.run_forever(speed_sp=self.CONST_SPEED)

        while not self.btn.any():  # While no button is pressed.
            left_color_intensity = self.color_sensor_left.value() / 100.
            right_color_intensity = self.color_sensor_right.value() / 100.

            # TODO: maybe remember prev predicted speed and use that
            curr_speed_left, curr_speed_right = self.motor_left.speed / self.L_MOTOR_BASE, self.motor_right.speed / self.R_MOTOR_BASE
            curr_speed_ratio = ((curr_speed_left + 1) / (curr_speed_right + 1) - 1) / 1000
            X = np.array([curr_speed_ratio, left_color_intensity, right_color_intensity])
            print("input: {:3.2f}\t{:1.3f}\t{:1.3f}".format(curr_speed_ratio,
                                                            left_color_intensity, right_color_intensity))
            start_time = time.time()
            speed_ratio = self.MLP.predict(X.reshape(1, -1))[0]
            speed_right = self.CONST_SPEED * speed_ratio

            print("output: {:1.2f}, r_speed {:3.2f}, prediction took {:3.3f}ms".format(speed_ratio, speed_right, (
                time.time() - start_time) * 1000))

            self.motor_right.run_forever(speed_sp=speed_right)
            time.sleep(0.01)

        self.motor_left.stop(stop_action="brake")
        self.motor_right.stop(stop_action="brake")

if __name__ == '__main__':
    print("Program started")
    model = LineFollowing()

    model_path = "model/double_straight_line_relu.p"

    model.calibration_color_sensor_values()
    # model.train("data/double_straight_line.csv", model_path)
    # model.run(model_path)