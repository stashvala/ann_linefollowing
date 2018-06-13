from time import time, sleep
from robot import Robot


class SampleCollector:

    SLICE = 10

    def __init__(self):
        self.robot = None

    def read_track(self, track_id=0):
        while not self.robot.btn.any():  # While no button is pressed.
            sleep(0.01)

        self.robot.reset_gyro()
        win_size = 3

        sleep(1)
        self.robot.beep()

        start_time = time()

        track_data = []
        while not self.robot.btn.any():  # While no button is pressed.
            angle = self.robot.gyroscope_value(win_size)

            left_motor_speed = self.robot.motor_left.speed
            right_motor_speed = self.robot.motor_right.speed

            left_color_intensity = self.robot.color_sensor_left.value()
            right_color_intensity = self.robot.color_sensor_right.value()

            t = (time() - start_time) * 1000  # miliseconds
            track_data.append(
                [track_id, angle, left_motor_speed, right_motor_speed,
                 left_color_intensity, right_color_intensity, t])

        self.robot.beep()

        assert(len(track_data) > self.SLICE * 2)
        return track_data[self.SLICE:-self.SLICE]  # slice array to remove beginning and ending that might contain noise

    @staticmethod
    def create_csv(file_name, data, header=None):
        if header is None:
            header = ["left_motor_speed", "right_motor_speed", "left_color_intensity", "right_color_intensity"]

        if len(header) != len(data[0]):
            print("Warning: header length and data length don't match!")

        with open(file_name, 'w') as csv_file:
            csv_file.write(",".join(header) + '\n')
            for line in data:
                formatted = ["{:4.3f}".format(i) if isinstance(i, float) else str(i) for i in line]
                str_line = ','.join(formatted) + '\n'
                csv_file.write(str_line)

        print("Total lines in csv = ", len(data))
        print("OUTPUT WRITTEN TO", file_name)

    def init_robot(self):
        if self.robot is None:
            self.robot = Robot()

        if not self.robot.is_ev3_set():
            self.robot.setup_ev3()

    def record_data(self):
        data = []
        index = 0

        self.init_robot()

        while True:
            print("PRESS A BUTTON TO RECORD NEW TRACK")
            x = self.read_track(index)
            data.extend(x)
            i = input('ONE MORE? [y/n] ')
            if i != 'y':
                break
            index += 1

        file_name = "data/" + input('CSV file: ') + '.csv'
        header = ["track_id", "angle", "left_motor_speed", "right_motor_speed", "left_color_intensity",
                  "right_color_intensity", "time"]

        self.create_csv(file_name, data, header)


if __name__ == '__main__':
    sc = SampleCollector()
    sc.record_data()

