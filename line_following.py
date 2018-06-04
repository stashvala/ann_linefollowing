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
# color_sensor_left = ColorSensor()
# assert color_sensor_left.connected, "Connect a color sensor to any sensor port"
#
# # color_sensor_right = ColorSensor('in3')
# # assert color_sensor_right.connected, "Connect a RIGHT color sensor to sensor port 3"
#
# color_sensor_left.mode = 'COL-REFLECT'
# # color_sensor_right.mode = 'COL-REFLECT'
#
# # BUTTON
# btn = Button()


class LineFollowing:

    def __init__(self, hidden_layer_sizes=(5, 5), learning_rate=0.0001, epochs=20000, batch_size=128):
        self.MLP = MLPRegressor(hidden_layer_sizes, learning_rate_init=learning_rate,
                                max_iter=epochs, batch_size=batch_size, verbose=True)
        pass

    def train(self, train_csv, model_path="model/mlp.p"):
        data = np.genfromtxt(train_csv, delimiter=',')

        X = data[1:, 4].reshape(-1, 1)  # remove if more features
        y = data[1:, [2, 3]]

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
    model.train("data/electrical_tape.csv")
    #model.run()