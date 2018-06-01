from ev3dev.ev3 import *
from time import time, sleep
import csv

# GYRO SENSOR
gyro_sensor = GyroSensor('in1')
assert gyro_sensor.connected, "Connect a gyro sensor to sensor port 1"
gyro_sensor.mode = 'GYRO-ANG'

# MOTORS
motor_left = LargeMotor('outB')
motor_right = LargeMotor('outD')

# COLOR SENSOR
color_sensor_left = ColorSensor('in2')
assert color_sensor_left.connected, "Connect a LEFT color sensor to sensor port 2"

color_sensor_right = ColorSensor('in3')
assert color_sensor_right.connected, "Connect a RIGHT color sensor to sensor port 3"

color_sensor_left.mode = 'COL-REFLECT'
color_sensor_right.mode = 'COL-REFLECT'

# BUTTON
btn = Button('in4')
assert btn.connected, "Connect a button to sensor port 4"

def read_track(id = 0):
    data = []
    while btn.any()==False: # While no button is pressed.
            sleep(0.01)  # Wait 0.01 second

    Sound.beep().wait()

    units = gs.units
    while btn.any()==False: # While no button is pressed.
        # GRYO ANGLE
        angle = gs.value()

        #MOTOR SPEED
        left_motor_counter_tacho = motor_left.count_per_rot
        right_motor_counter_tacho = motor_right.count_per_rot

        left_motor_speed = motor_left.speed
        right_motor_speed = motor_right.speed

        # COLOR SENSOR LIGHT INTENSITY
        left_color_intensity =color_sensor_left.value()
        right_color_intensity = color_sensor_right.value()

        data.append((id, angle, units, left_motor_counter_tacho, right_motor_counter_tacho, left_motor_speed, right_motor_speed, left_color_intensity, right_color_intensity))
    return data


data = []
index = 1

# RECORD DATA
while True:
    print("PRESS BUTTON TO RECORD NEW TRACK")
    x = read_track(index)
    data.append(x)
    i = input('ONE MORE? [y/n] ')
    if i != 'y':
        break
    index += 1

# TO CSV
fileName = input('CSV file: ')
with open(fileName, 'wb') as csv_file:
    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
    wr.writerow(data)
    