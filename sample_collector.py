from ev3dev.ev3 import *
from time import time, sleep
import csv

# GYRO SENSOR
gs = GyroSensor()
assert gs.connected, "Connect a gyro sensor to any sensor port"
gs.mode = 'GYRO-ANG'

# MOTORS
motor_left = LargeMotor('outB')
motor_right = LargeMotor('outD')

# COLOR SENSOR
color_sensor_left = ColorSensor()
assert color_sensor_left.connected, "Connect a color sensor to any sensor port"

color_sensor_right = ColorSensor()
assert color_sensor_right.connected, "Connect a RIGHT color sensor to sensor port 3"

color_sensor_left.mode = 'COL-REFLECT'
color_sensor_right.mode = 'COL-REFLECT'

# BUTTON
btn = Button()

# assert btn.connected, "Connect a button to sensor port 4"


def reset_gyro():
    # RESET GYRO
    gs.mode = 'GYRO-RATE'
    gs.mode = 'GYRO-ANG'


def gyroscope_value(w_size):
    return sum([gs.value() for _ in range(w_size)]) / w_size


def read_track(id=0):
    while not btn.any():  # While no button is pressed.
        sleep(0.01)

    reset_gyro()
    win_size = 3

    sleep(1)
    Sound.beep().wait()

    track_data = []
    while not btn.any():  # While no button is pressed.
        # GRYO ANGLE
        angle = gyroscope_value(win_size)

        # MOTOR SPEED
        # left_motor_counter_tacho = motor_left.count_per_rot
        # right_motor_counter_tacho = motor_right.count_per_rot

        left_motor_speed = motor_left.speed
        right_motor_speed = motor_right.speed

        # COLOR SENSOR LIGHT INTENSITY
        left_color_intensity = color_sensor_left.value()
        right_color_intensity = color_sensor_right.value()

        track_data.append(
            [id, angle, left_motor_speed, right_motor_speed,
             left_color_intensity, right_color_intensity])

    Sound.beep().wait()

    return track_data


data = []
index = 0

# RECORD DATA
while True:
    print("PRESS A BUTTON TO RECORD NEW TRACK")
    x = read_track(index)
    data.extend(x)
    i = input('ONE MORE? [y/n] ')
    if i != 'y':
        break
    index += 1

# WRITE TO CSV
file_name = "data/" + input('CSV file: ') + '.csv'
with open(file_name, 'w') as csv_file:
    header = ["track_id", "angle", "left_motor_speed", "right_motor_speed", "left_color_intensity", "right_color_intensity"]
    if len(header) != len(data[0]):
        print("Warning: header length and data length don't match!")

    csv_file.write(",".join(header) + '\n')
    for line in data:
        formatted = ["{:3.2f}".format(i) if isinstance(i, float) else str(i) for i in line]
        str_line = ','.join(formatted) + '\n'
        csv_file.write(str_line)

print("OUTPUT WRITTEN TO", file_name)
