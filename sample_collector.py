from ev3dev.ev3 import *
from time import time, sleep

# GYRO SENSOR
gs = GyroSensor()
assert gs.connected, "Connect a gyro sensor to any sensor port"
gs.mode = 'GYRO-ANG'

# MOTORS
motor_left = LargeMotor('outB')
motor_right = LargeMotor('outC')

# COLOR SENSOR
color_sensor_left = ColorSensor('in1')
assert color_sensor_left.connected, "Connect LEFT color sensor to sensor port 1"

color_sensor_right = ColorSensor('in2')
assert color_sensor_right.connected, "Connect RIGHT color sensor to sensor port 2"

color_sensor_left.mode = 'COL-REFLECT'
color_sensor_right.mode = 'COL-REFLECT'

# BUTTON
btn = Button()

# assert btn.connected, "Connect a button to sensor port 4"

SLICE = 10


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

    assert(len(track_data) > SLICE * 2)
    return track_data[SLICE:-SLICE]  # slice array to remove beginning and ending that might contain noise


def create_csv(file_name, data, header=None):
    if header is None:
        header = ["left_motor_speed", "right_motor_speed", "left_color_intensity", "right_color_intensity"]

    if len(header) != len(data[0]):
        print("Warning: header length and data length don't match!")

    with open(file_name, 'w') as csv_file:
        csv_file.write(",".join(header) + '\n')
        for line in data:
            formatted = ["{:3.2f}".format(i) if isinstance(i, float) else str(i) for i in line]
            str_line = ','.join(formatted) + '\n'
            csv_file.write(str_line)

    print("Total lines in csv = ", len(data))
    print("OUTPUT WRITTEN TO", file_name)


def record_data():
    data = []
    index = 0

    while True:
        print("PRESS A BUTTON TO RECORD NEW TRACK")
        x = read_track(index)
        data.extend(x)
        i = input('ONE MORE? [y/n] ')
        if i != 'y':
            break
        index += 1

    file_name = "data/" + input('CSV file: ') + '.csv'
    header = ["track_id", "angle", "left_motor_speed", "right_motor_speed", "left_color_intensity",
              "right_color_intensity"]

    create_csv(file_name, data, header)


if __name__ == '__main__':
    record_data()

