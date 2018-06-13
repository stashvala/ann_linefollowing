from ev3dev.ev3 import *


class Robot:

    def __init__(self):
        self.gs = None
        self.motor_left = None
        self.motor_right = None
        self.color_sensor_left = None
        self.color_sensor_right = None
        self.btn = None

    def setup_ev3(self):
        print("Setting up ev3")

        self.gs = GyroSensor()
        assert self.gs.connected, "Connect a gyro sensor to any sensor port"
        self.gs.mode = 'GYRO-ANG'

        self.motor_left = LargeMotor('outB')
        self.motor_right = LargeMotor('outC')

        self.color_sensor_left = ColorSensor('in1')
        assert self.color_sensor_left.connected, "Connect LEFT color sensor to sensor port 1"

        self.color_sensor_right = ColorSensor('in2')
        assert self.color_sensor_right.connected, "Connect RIGHT color sensor to sensor port 2"

        self.color_sensor_left.mode = 'COL-REFLECT'
        self.color_sensor_right.mode = 'COL-REFLECT'

        self.btn = Button()

    def is_ev3_set(self):
        return self.motor_right is not None and self.motor_left is not None and \
               self.color_sensor_right is not None and self.color_sensor_right is not None

    def drive(self, left_motor_speed, right_motor_speed):
        self.motor_left.run_forever(speed_sp=left_motor_speed)
        self.motor_right.run_forever(speed_sp=right_motor_speed)

    def stop(self):
        self.motor_left.stop(stop_action="brake")
        self.motor_right.stop(stop_action="brake")

    def gyroscope_value(self, w_size):
        return sum([self.gs.value() for _ in range(w_size)]) / w_size

    def reset_gyro(self):
        self.gs.mode = 'GYRO-RATE'
        self.gs.mode = 'GYRO-ANG'

    @staticmethod
    def beep():
        Sound.beep().wait()
