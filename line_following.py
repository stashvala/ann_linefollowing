import pickle

import numpy as np
from sklearn.neural_network import MLPRegressor
from time import sleep
# from ev3dev.ev3 import *

# GYRO SENSOR
# gs = GyroSensor()
# assert gs.connected, "Connect a gyro sensor to any sensor port"
# gs.mode = 'GYRO-ANG'
#
# # MOTORS
# motor_left = LargeMotor('outB')
# motor_right = LargeMotor('outD')
#
# # COLOR SENSOR
# color_sensor_left = ColorSensor('in1')
# assert color_sensor_left.connected, "Connect LEFT color sensor to sensor port 1"
#
# color_sensor_right = ColorSensor('in2')
# assert color_sensor_right.connected, "Connect RIGHT color sensor to sensor port 2"
#
# color_sensor_left.mode = 'COL-REFLECT'
# # color_sensor_right.mode = 'COL-REFLECT'
#
# # BUTTON
# btn = Button()


class LineFollowing:

    MOTORS = [2, 3]
    COLOR_SENSORS = [4, 5]

    def __init__(self, hidden_layer_sizes=(100, ), solver="lbfgs", learning_rate=0.0001, epochs=20000, batch_size=256):
        self.MLP = MLPRegressor(hidden_layer_sizes, solver=solver, learning_rate_init=learning_rate, shuffle=False,
                                activation="logistic", max_iter=epochs, batch_size=batch_size, verbose=True)
        pass

    def train(self, train_csv, model_path="model/mlp.p"):
        data = np.genfromtxt(train_csv, delimiter=',', skip_header=1)

        X = data[:, self.COLOR_SENSORS]
        X /= 100  # normalize reflected color
        y = data[:, self.MOTORS]

        data = None  # free memory if python wills it

        self.MLP.fit(X, y)

        pickle.dump(self.MLP, open(model_path, "wb"))

    def run(self, model_path=None):
        if model_path is not None:
            self.MLP = pickle.load(open(model_path, "rb"))

        while not btn.any():  # While no button is pressed.
            left_color_intensity = color_sensor_left.value()

            X = np.array([left_color_intensity])

            speed_left, speed_right = self.MLP.predict(X)

            motor_left.run_forever(speed_left)
            motor_right.run_forever(speed_right)
            sleep(0.01)


if __name__ == '__main__':
    model = LineFollowing()
    # model.train("data/electrical_tape.csv")
    model.run("model/mlp")