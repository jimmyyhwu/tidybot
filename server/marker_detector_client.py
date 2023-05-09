# Author: Jimmy Wu
# Date: February 2023

import argparse
import math
import time
from multiprocessing.connection import Client
import matplotlib.pyplot as plt
import numpy as np
from constants import NUM_FLOOR_TILES_X, NUM_FLOOR_TILES_Y, FLOOR_TILE_SIZE, FLOOR_LENGTH, FLOOR_WIDTH, NUM_ROBOTS, ROBOT_WIDTH

class RobotVisualizer:
    def __init__(self, robot_idx):
        self.robot_num = robot_idx + 1
        plt.ion()
        plt.figure('Robots')
        plt.axis([-(FLOOR_WIDTH / 2), FLOOR_WIDTH / 2, -(FLOOR_LENGTH / 2), FLOOR_LENGTH / 2])
        plt.gca().set_aspect('equal')
        plt.xticks(FLOOR_TILE_SIZE * (-(NUM_FLOOR_TILES_X / 2) + np.arange(NUM_FLOOR_TILES_X + 1)))
        plt.yticks(FLOOR_TILE_SIZE * (-(NUM_FLOOR_TILES_Y / 2) + np.arange(NUM_FLOOR_TILES_Y + 1)))
        plt.grid(which='both')
        self.robot_line, = plt.plot([], color='tab:gray')
        self.robot_arrow = plt.arrow(0, 0, 0, 0, head_width=0)
        self.robot_text = plt.text(0, 0, None, ha='center', va='center')

    def draw(self, pose):
        if pose is None:
            self.robot_line.set_data([], [])
            self.robot_arrow.set_data(x=0, y=0, dx=0, dy=0, head_width=0)
            self.robot_text.set_text(None)
            return

        # Draw robot outline
        heading = pose[2]
        angles = heading + np.radians([135, 45, -45, -135], dtype=np.float32)
        corners = (math.sqrt(2) / 2) * np.stack((np.cos(angles, dtype=np.float32), np.sin(angles, dtype=np.float32)), axis=1)
        corners = np.array([pose[0], pose[1]], dtype=np.float32) + ROBOT_WIDTH * corners
        corners = np.append(corners, corners[:1], axis=0)
        self.robot_line.set_data(*corners.T)

        # Draw robot heading
        arrow_dx = 0.25 * ROBOT_WIDTH * math.cos(heading)
        arrow_dy = 0.25 * ROBOT_WIDTH * math.sin(heading)
        self.robot_arrow.set_data(x=pose[0], y=pose[1], dx=arrow_dx, dy=arrow_dy, head_width=ROBOT_WIDTH / 8)

        # Label robot number
        self.robot_text.set_position((pose[0] - (ROBOT_WIDTH / 4) * math.cos(heading), pose[1] - (ROBOT_WIDTH / 4) * math.sin(heading)))
        self.robot_text.set_text(self.robot_num)

class DebugVisualizer:
    def __init__(self):
        plt.figure(1)
        self.debug_lines = [plt.plot([], color='tab:orange')[0] for _ in range(4)]
        self.debug_texts = [plt.text(0, 0, None, ha='center', va='center') for _ in range(4)]

    def draw_debug_data(self, debug_data):
        for i in range(4):
            if i > len(debug_data) - 1:
                self.debug_lines[i].set_data([], [])
                self.debug_texts[i].set_text(None)
            else:
                self.debug_lines[i].set_data([debug_data[i][1][0], debug_data[i][2][0]], [debug_data[i][1][1], debug_data[i][2][1]])
                self.debug_texts[i].set_text(debug_data[i][0])
                self.debug_texts[i].set_position((debug_data[i][1][0], debug_data[i][1][1]))
        if len(debug_data) > 4:
            print(f'Warning: Found {len(debug_data)} debug lines but only first 4 will be drawn.')

def main(args):
    conn = Client(('localhost', 6002), authkey=b'secret password')
    visualizers = [RobotVisualizer(robot_idx) for robot_idx in range(NUM_ROBOTS)]
    if args.debug:
        debug_visualizer = DebugVisualizer()

    while True:
        if args.benchmark:
            start_time = time.time()

        conn.send(None)
        data = conn.recv()
        for robot_idx in range(NUM_ROBOTS):
            pose = data['poses'].get(robot_idx, None)
            visualizers[robot_idx].draw(pose)
            if args.debug and 'debug_data' in data:
                debug_visualizer.draw_debug_data(data['debug_data'])
        plt.pause(0.001)  # 15 ms

        if args.benchmark:
            print(f'{1000 * (time.time() - start_time):.1f} ms')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
