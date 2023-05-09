# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from multiprocessing import Process
from multiprocessing.connection import Listener
from queue import Queue
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
from camera_server import CameraServer
from marker_detector_server import MarkerDetectorServer
from marker_detector_client import RobotVisualizer

class TrajectoryVisualizer(RobotVisualizer):
    def __init__(self, robot_idx):
        super().__init__(robot_idx)
        plt.figure('Robots').canvas.manager.window.wm_geometry('+800+0')
        plt.figure('Robots').canvas.manager.set_window_title(f'Robot {robot_idx + 1}')
        self.waypoints_line, = plt.plot([], color='tab:green')
        self.target_ee_pos_line, = plt.plot([], marker='o', color='tab:red')
        self.traj_map_line, = plt.plot([], color='tab:blue')
        self.traj_odom_line, = plt.plot([], '--', color='tab:blue')

    def draw(self, robot_pose, waypoints, target_ee_pos, traj_map, traj_odom):  # pylint: disable=arguments-differ
        super().draw(robot_pose)

        # Waypoints
        if len(waypoints) > 0:
            waypoints = np.array(waypoints, dtype=np.float32)
            self.waypoints_line.set_data(waypoints[:, 0], waypoints[:, 1])
        else:
            self.waypoints_line.set_data([], [])

        # Target end effector position
        if target_ee_pos is not None:
            self.target_ee_pos_line.set_data(*target_ee_pos)
        else:
            self.target_ee_pos_line.set_data([], [])

        # Trajectory (map)
        if len(traj_map) > 0:
            traj_map = np.array(traj_map, dtype=np.float32)
            self.traj_map_line.set_data(traj_map[:, 0], traj_map[:, 1])
        else:
            self.traj_map_line.set_data([], [])

        # Trajectory (odometry)
        if len(traj_odom) > 0:
            traj_odom = np.array(traj_odom, dtype=np.float32)
            self.traj_odom_line.set_data(traj_odom[:, 0], traj_odom[:, 1])
        else:
            self.traj_odom_line.set_data([], [])

class DebugVisualizer(TrajectoryVisualizer):
    def __init__(self, robot_idx):
        super().__init__(robot_idx)
        plt.figure('Robots')
        self.lookahead_line, = plt.plot([], marker='o', color='tab:red')
        plt.figure('Velocities').canvas.manager.window.wm_geometry('+800+600')
        self.vel_fig_axis = plt.gca()
        self.vel_x_line, = plt.plot([], label='x')
        self.vel_y_line, = plt.plot([], label='y')
        self.vel_th_line, = plt.plot([], label='θ')
        plt.grid(True)
        plt.legend()

    def draw(self, robot_pose, waypoints, target_ee_pos, traj_map, traj_odom, lookahead_pos, velocities):  # pylint: disable=arguments-differ
        super().draw(robot_pose, waypoints, target_ee_pos, traj_map, traj_odom)

        # Lookahead position
        if lookahead_pos is not None:
            self.lookahead_line.set_data([robot_pose[0], lookahead_pos[0]], [robot_pose[1], lookahead_pos[1]])
        else:
            self.lookahead_line.set_data([], [])

        # Velocities
        if len(velocities) > 0:
            velocities = np.array(velocities, dtype=np.float64)  # Do not cast timestamps to np.float32, major loss of precision
            t = velocities[:, 0]
            self.vel_x_line.set_data(t, velocities[:, 1])
            self.vel_y_line.set_data(t, velocities[:, 2])
            self.vel_th_line.set_data(t, velocities[:, 3])
            self.vel_fig_axis.relim()
            self.vel_fig_axis.autoscale()
        else:
            self.vel_x_line.set_data([], [])
            self.vel_y_line.set_data([], [])
            self.vel_th_line.set_data([], [])
            self.vel_fig_axis.set_xlim(-0.055, 0.055)
            self.vel_fig_axis.set_ylim(-0.055, 0.055)

class ControllerServer:
    def __init__(self, robot_idx, debug=False):
        self.controller_data = {'state': 'idle'}
        self.base_pose = (0, 0, 0)

        # Connections to client and robot
        self.client_listener = Listener(('0.0.0.0', 6004 + robot_idx), authkey=b'secret password')
        self.robot_listener = Listener(('0.0.0.0', 6007 + robot_idx), authkey=b'secret password')

        # Queue for passing commands from client to the robot
        self.queue = Queue(maxsize=1)

        # Visualization
        self.waypoints = []
        self.target_ee_pos = None
        self.traj_map = []
        self.traj_odom = []
        self.debug = debug
        if self.debug:
            self.velocities = []
            self.traj_visualizer = DebugVisualizer(robot_idx)
        else:
            self.traj_visualizer = TrajectoryVisualizer(robot_idx)

    def reset_visualizer(self):
        self.waypoints = []
        self.target_ee_pos = None
        self.traj_map = []
        self.traj_odom = []
        if self.debug:
            self.velocities = []

    def handle_client_conn(self):
        while True:
            # Connect to client
            address, port = self.client_listener.address
            print(f'Waiting for client connection ({address}:{port})')
            conn = self.client_listener.accept()
            print('Connected to client!')

            try:
                last_time = time.time()
                while True:
                    while time.time() - last_time < 0.0333:  # 30 Hz
                        time.sleep(0.0001)
                    step_time = time.time() - last_time
                    if step_time > 0.04:  # 25 Hz
                        print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__} handle_client_conn')
                    last_time = time.time()

                    if conn.poll():
                        command = conn.recv()
                        if command == 'controller_data':
                            conn.send(self.controller_data)
                        elif command == 'reset_visualizer':
                            self.reset_visualizer()
                        else:
                            while not self.queue.empty():
                                time.sleep(0.0001)
                            self.queue.put(command)
                            if command != 'stop':
                                self.reset_visualizer()

            except (ConnectionResetError, EOFError, BrokenPipeError):
                pass

    def run(self):
        # Start separate thread to handle client conn
        Thread(target=self.handle_client_conn, daemon=True).start()

        # Handle robot conn
        while True:
            # Connect to robot
            address, port = self.robot_listener.address
            print(f'Waiting for robot connection ({address}:{port})')
            conn = self.robot_listener.accept()
            print('Connected to robot!')

            try:
                last_time = time.time()
                while True:
                    step_time = time.time() - last_time
                    if step_time > 0.15:  # 6 Hz
                        print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__}')
                    last_time = time.time()

                    # Communicate with controller
                    controller_data = conn.recv()
                    conn.send(None if self.queue.empty() else self.queue.get())

                    # Store controller data
                    self.controller_data = controller_data
                    robot_state = self.controller_data['state']
                    if 'base' in self.controller_data:
                        self.base_pose = self.controller_data['base']['pose']
                        base_position = (self.base_pose[0], self.base_pose[1])
                        if robot_state == 'moving':
                            self.waypoints = self.controller_data['base']['waypoints']
                            self.target_ee_pos = self.controller_data['base']['target_ee_pos']
                            self.traj_map.append(base_position)
                            self.traj_odom.append((self.controller_data['base']['pose_odom'][0], self.controller_data['base']['pose_odom'][1]))
                    if 'arm' in self.controller_data:
                        if robot_state == 'manipulating':
                            self.target_ee_pos = self.controller_data['arm']['target_ee_pos']
                    if self.debug:
                        lookahead_pos = None
                        if 'base' in self.controller_data and robot_state == 'moving':
                            lookahead_pos = self.controller_data['base'].get('lookahead_position', None)
                            if 'velocity' in self.controller_data['base']:
                                self.velocities.append(self.controller_data['base']['velocity'])

                    # Update visualizer
                    if self.debug:
                        self.traj_visualizer.draw(self.base_pose, self.waypoints, self.target_ee_pos, self.traj_map, self.traj_odom, lookahead_pos, self.velocities)
                    else:
                        self.traj_visualizer.draw(self.base_pose, self.waypoints, self.target_ee_pos, self.traj_map, self.traj_odom)  # pylint: disable=no-value-for-parameter
                    plt.pause(0.001)  # 15 ms

            except (ConnectionResetError, EOFError, BrokenPipeError):
                pass

def main(args):
    # Start camera servers
    def start_camera_server(serial, port):
        CameraServer(serial, port=port).run()
    for serial, port in [('E4298F4E', 6000), ('099A11EE', 6001)]:
        Process(target=start_camera_server, args=(serial, port), daemon=True).start()

    # Wait for camera servers to be ready
    time.sleep(1.5)

    # Start marker detector server
    def start_marker_detector_server():
        MarkerDetectorServer(hostname='0.0.0.0').run()
    Process(target=start_marker_detector_server, daemon=True).start()

    # Start controller server
    robot_idx = args.robot_num - 1
    ControllerServer(robot_idx, debug=args.debug).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-num', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
