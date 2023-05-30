# Author: Jimmy Wu
# Date: February 2023

import argparse
from multiprocessing.connection import Client
from constants import CONN_AUTHKEY

class ControllerClient:
    def __init__(self, robot_idx):
        self.conn = Client(('localhost', 6004 + robot_idx), authkey=CONN_AUTHKEY)

    def get_controller_data(self):
        self.conn.send('controller_data')
        return self.conn.recv()

    def execute_command(self, command):
        self.conn.send(command)

    def stop(self):
        self.conn.send('stop')

    def reset_visualizer(self):
        self.conn.send('reset_visualizer')

def main(args):
    robot_idx = args.robot_num - 1
    client = ControllerClient(robot_idx)
    client.execute_command({
        'primitive_name': 'move',
        'waypoints': [(-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)],
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-num', type=int, default=1)
    main(parser.parse_args())
