# Author: Jimmy Wu
# Date: February 2023

import argparse
import logging
import math
import random
import socket
import subprocess
import threading
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path
from queue import Queue
from kortex_api.Exceptions.KServerException import KServerException
from redis import Redis
from camera import Camera
from constants import SERVER_HOSTNAME, ROBOT_HOSTNAME_PREFIX, CONN_AUTHKEY, REDIS_PASSWORD
from constants import ARM_HEADING_COMPENSATION
from kinova import KinovaArm

def distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def restrict_heading_range(h):
    return (h + math.pi) % (2 * math.pi) - math.pi

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def intersect(d, f, r, use_t1=False):
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
    a = dot(d, d)
    b = 2 * dot(f, d)
    c = dot(f, f) - r * r
    discriminant = (b * b) - (4 * a * c)
    if discriminant >= 0:
        if use_t1:
            t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
            if 0 <= t1 <= 1:
                return t1
        else:
            t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
            if 0 <= t2 <= 1:
                return t2
    return None

class RedisClient:
    def __init__(self):
        hostname = socket.gethostname()
        assert hostname.startswith(ROBOT_HOSTNAME_PREFIX)
        self.bot_num = int(hostname[-1])
        self.client = Redis(f'{ROBOT_HOSTNAME_PREFIX}{self.bot_num}', password=REDIS_PASSWORD, decode_responses=True)

    def get_driver_version(self):
        redis_key = f'mmp::bot{self.bot_num}::veh::driver_version'
        self.client.delete(redis_key)
        time.sleep(3 * 0.008)  # 3 cycles at 125 Hz
        return self.client.get(redis_key)

    def get_pose(self):
        return tuple(map(float, self.client.get(f'mmp::bot{self.bot_num}::veh::sensor::x').split(' ')))

    def set_target_pose(self, pose):
        self.client.set(f'mmp::bot{self.bot_num}::veh::control::x', f'{pose[0]} {pose[1]} {pose[2]}')

    def get_goal_reached(self):
        return bool(int(self.client.get(f'mmp::bot{self.bot_num}::veh::sensor::goal_reached')))

    def set_stop(self, value):
        self.client.set(f'mmp::bot{self.bot_num}::veh::stop', int(value))

    def set_max_velocity(self, max_vel_x, max_vel_y, max_vel_theta):
        self.client.set(f'mmp::bot{self.bot_num}::veh::control::max_vel', f'{max_vel_x} {max_vel_y} {max_vel_theta}')

    def set_max_acceleration(self, max_accel_x, max_accel_y, max_accel_theta):
        self.client.set(f'mmp::bot{self.bot_num}::veh::control::max_accel', f'{max_accel_x} {max_accel_y} {max_accel_theta}')

    def get_velocity(self):
        return tuple(map(float, self.client.get(f'mmp::bot{self.bot_num}::veh::sensor::dx').split(' ')))

    def get_cstop(self):
        return bool(int(self.client.get('mmp::cstop')))

    def get_emergency_shutdown(self):
        return bool(int(self.client.get('mmp::emergency_shutdown')))

class CoordFrameConverter:
    def __init__(self, pose_in_map, pose_in_odom):
        self.origin = None
        self.basis = None
        self.update(pose_in_map, pose_in_odom)

    def update(self, pose_in_map, pose_in_odom):
        self.basis = pose_in_map[2] - pose_in_odom[2]
        dx = pose_in_odom[0] * math.cos(self.basis) - pose_in_odom[1] * math.sin(self.basis)
        dy = pose_in_odom[0] * math.sin(self.basis) + pose_in_odom[1] * math.cos(self.basis)
        self.origin = (pose_in_map[0] - dx, pose_in_map[1] - dy)

    def convert_position(self, position):
        x, y = position
        x = x - self.origin[0]
        y = y - self.origin[1]
        xp = x * math.cos(-self.basis) - y * math.sin(-self.basis)  # pylint: disable=invalid-unary-operand-type
        yp = x * math.sin(-self.basis) + y * math.cos(-self.basis)  # pylint: disable=invalid-unary-operand-type
        return (xp, yp)

    def convert_heading(self, th):
        return th - self.basis

    def convert_pose(self, pose):
        x, y, th = pose
        return (*self.convert_position((x, y)), self.convert_heading(th))

class BaseController:
    LOOKAHEAD_DISTANCE = 0.3  # 30 cm

    def __init__(self, debug=False):
        # Mobile base driver
        self.driver_running = True
        ps_output = subprocess.run(['ps', '-C', 'vehicle', 'H', '-o', 'rtprio='], capture_output=True, text=True).stdout  # pylint: disable=subprocess-run-check
        if '80' not in map(str.strip, ps_output.split('\n')):  # Control thread runs with rtprio 80
            self.driver_running = False
            print('Mobile base driver is not running, please restart this controller after the driver is ready')
        self.redis_client = RedisClient()
        self.redis_client.set_stop(True)
        self.redis_client.set_max_velocity(0.5, 0.5, 3.14)
        self.redis_client.set_max_acceleration(0.5, 0.5, 2.36)
        self.robot_idx = self.redis_client.bot_num - 1
        if self.driver_running:
            expected_driver_version = '2023-08-17'
            assert self.redis_client.get_driver_version() == expected_driver_version, f'Please make sure you are running the correct version of the mobile base driver ({expected_driver_version})'

        # Control loop
        self.running = False
        self.state = 'idle'  # States: idle, moving
        self.pose_map = (0, 0, 0)
        self.pose_odom = self.redis_client.get_pose()
        self.map_to_odom_converter = CoordFrameConverter(self.pose_map, self.pose_odom)
        self.odom_to_map_converter = CoordFrameConverter(self.pose_odom, self.pose_map)
        self.waypoints_map = None
        self.waypoints_odom = None
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.target_ee_pos_map = None
        self.target_ee_pos_odom = None
        self.lookahead_position_odom = None
        self.position_tolerance = None
        self.heading_tolerance = None

        # Checking for excessive drift
        self.excessive_position_drift_count = 0
        self.excessive_heading_drift_count = 0

        self.debug = debug
        if self.debug:
            self.odom_to_map_converter_lookahead = CoordFrameConverter(self.pose_odom, self.pose_map)

    def execute_command(self, base_command):
        self.map_to_odom_converter.update(self.pose_map, self.pose_odom)
        self.odom_to_map_converter.update(self.pose_odom, self.pose_map)
        self.waypoints_map = base_command['waypoints']
        self.waypoints_odom = list(map(self.map_to_odom_converter.convert_position, self.waypoints_map))
        self.waypoint_index = 1
        self.target_ee_pos_map = base_command['target_ee_pos']
        self.target_ee_pos_odom = None if self.target_ee_pos_map is None else self.map_to_odom_converter.convert_position(self.target_ee_pos_map)
        self.lookahead_position_odom = None
        self.position_tolerance = base_command.get('position_tolerance', 0.015)  # Default 1.5 cm
        self.heading_tolerance = base_command.get('heading_tolerance', math.radians(2.1))  # Default 2.1 deg
        if self.driver_running:
            self.redis_client.set_stop(False)
            self.state = 'moving'

    def stop(self):
        self.redis_client.set_stop(True)
        self.state = 'idle'

    def get_controller_data(self):
        controller_data = {
            'state': self.state,
            'pose': self.pose_map,
            'pose_odom': self.odom_to_map_converter.convert_pose(self.pose_odom),
            'waypoints': self.waypoints_map,
            'target_ee_pos': self.target_ee_pos_map,
        }
        if self.debug:
            velocity = self.redis_client.get_velocity()
            controller_data['velocity'] = (time.time(), velocity[0], velocity[1], velocity[2])
            if self.lookahead_position_odom is not None:
                self.odom_to_map_converter_lookahead.update(self.pose_odom, self.pose_map)
                controller_data['lookahead_position_odom'] = self.odom_to_map_converter_lookahead.convert_position(self.lookahead_position_odom)
        return controller_data

    def run(self):
        try:
            self.running = True

            # Robot pose from marker detection
            marker_detector_conn = Client((SERVER_HOSTNAME, 6002), authkey=CONN_AUTHKEY)
            marker_detector_conn.send(None)

            goal_reached_steps = 0
            last_time = time.time()
            while True:
                while time.time() - last_time < 0.008:  # 125 Hz
                    time.sleep(0.0001)
                step_time = time.time() - last_time
                if step_time > 0.012:  # 83 Hz
                    print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__}')
                last_time = time.time()

                # Check for cstop or emergency shutdown
                if self.redis_client.get_cstop():
                    print('Exiting since cstop key was set')
                    break
                if self.redis_client.get_emergency_shutdown():
                    print('Exiting since emergency shutdown key was set')
                    break

                # Update pose from odometry and marker detection
                self.pose_odom = self.redis_client.get_pose()
                if marker_detector_conn.poll():
                    detector_data = marker_detector_conn.recv()
                    marker_detector_conn.send(None)
                    if self.robot_idx in detector_data['poses']:
                        self.pose_map = detector_data['poses'][self.robot_idx]

                        # Check for excessive drift
                        if self.state == 'moving':
                            pose_odom = self.odom_to_map_converter.convert_pose(self.pose_odom)

                            # Position drift
                            position_drift = distance(pose_odom, self.pose_map)
                            if position_drift > 0.3:  # 30 cm
                                self.excessive_position_drift_count += 1
                            else:
                                self.excessive_position_drift_count = 0
                            if self.excessive_position_drift_count > 4:  # 40 ms
                                print(f'Exiting due to excessive position drift ({100 * position_drift:.2f} cm)')
                                self.redis_client.set_stop(True)
                                break

                            # Heading drift
                            heading_drift = abs(restrict_heading_range(self.pose_map[2] - pose_odom[2]))
                            if heading_drift > math.radians(30):  # 30 deg
                                self.excessive_heading_drift_count += 1
                            else:
                                self.excessive_heading_drift_count = 0
                            if self.excessive_heading_drift_count > 4:  # 40 ms
                                print(f'Exiting due to excessive heading drift ({math.degrees(heading_drift):.2f} deg)')
                                self.redis_client.set_stop(True)
                                break

                # Base control logic
                if self.state == 'idle':
                    self.redis_client.set_target_pose(self.pose_odom)
                elif self.state == 'moving':
                    if self.redis_client.get_goal_reached():
                        goal_reached_steps += 1
                    if goal_reached_steps > 1:  # 16 ms
                        goal_reached_steps = 0
                        position_error = distance(self.pose_map, self.waypoints_map[-1])
                        if self.target_ee_pos_map is not None:
                            dx = self.target_ee_pos_map[0] - self.pose_map[0]
                            dy = self.target_ee_pos_map[1] - self.pose_map[1]
                            heading_error = abs(restrict_heading_range(math.atan2(dy, dx) + math.pi - self.pose_map[2]))
                        else:
                            heading_error = 0
                        if position_error > self.position_tolerance or heading_error > self.heading_tolerance:
                            # Execute corrective movement if robot is too far from intended destination
                            print(f'Too far from target pose ({100 * position_error:.2f} cm, {math.degrees(heading_error):.2f} deg)')
                            self.map_to_odom_converter.update(self.pose_map, self.pose_odom)
                            self.odom_to_map_converter.update(self.pose_odom, self.pose_map)
                            self.waypoints_odom = list(map(self.map_to_odom_converter.convert_position, self.waypoints_map))
                            self.target_ee_pos_odom = None if self.target_ee_pos_map is None else self.map_to_odom_converter.convert_position(self.target_ee_pos_map)
                        else:
                            self.redis_client.set_stop(True)
                            self.state = 'idle'
                    else:
                        # Compute lookahead position
                        while True:
                            start = self.waypoints_odom[self.waypoint_index - 1]
                            end = self.waypoints_odom[self.waypoint_index]
                            d = (end[0] - start[0], end[1] - start[1])
                            f = (start[0] - self.pose_odom[0], start[1] - self.pose_odom[1])
                            t2 = intersect(d, f, BaseController.LOOKAHEAD_DISTANCE)
                            if t2 is not None:
                                self.lookahead_position_odom = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                                break
                            if self.waypoint_index == len(self.waypoints_odom) - 1:
                                self.lookahead_position_odom = None
                                break
                            self.waypoint_index += 1
                        if self.lookahead_position_odom is None:
                            target_position = self.waypoints_odom[-1]
                        else:
                            target_position = self.lookahead_position_odom

                        # Compute target heading
                        target_heading = self.pose_odom[2]
                        if self.target_ee_pos_odom is not None:
                            # Turn to face target end effector position
                            dx = self.target_ee_pos_odom[0] - self.pose_odom[0]
                            dy = self.target_ee_pos_odom[1] - self.pose_odom[1]

                            frac = 1
                            if self.lookahead_position_odom is not None:
                                # Turn slowly at first, and then more quickly as robot approaches target
                                remaining_path_length = BaseController.LOOKAHEAD_DISTANCE
                                curr_waypoint = self.lookahead_position_odom
                                for idx in range(self.waypoint_index, len(self.waypoints_odom)):
                                    next_waypoint = self.waypoints_odom[idx]
                                    remaining_path_length += distance(curr_waypoint, next_waypoint)
                                    curr_waypoint = next_waypoint
                                frac = math.sqrt(BaseController.LOOKAHEAD_DISTANCE / remaining_path_length)
                            target_heading += frac * restrict_heading_range(math.atan2(dy, dx) + math.pi - self.pose_odom[2])

                        self.redis_client.set_target_pose((target_position[0], target_position[1], target_heading))

        finally:
            self.running = False
            print('Stopping mobile base movement before exiting')
            self.redis_client.set_stop(True)

class DummyBaseController:
    def __init__(self):
        self.state = 'idle'  # States: idle, moving
        self.pose_map = (0, 0, 0)
        self.pose_odom = (0, 0, 0)
        self.waypoints = None
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.target_ee_pos = None
        self.running = False

    def execute_command(self, base_command):
        self.waypoints = base_command['waypoints']
        self.waypoint_index = 1
        self.target_ee_pos = base_command['target_ee_pos']
        self.pose_odom = self.pose_map
        self.state = 'moving'

    def stop(self):
        self.state = 'idle'

    def get_controller_data(self):
        return {
            'state': self.state,
            'pose': self.pose_map,
            'pose_odom': self.pose_odom,
            'waypoints': self.waypoints,
            'target_ee_pos': self.target_ee_pos,
        }

    def run(self):
        try:
            self.running = True
            drift = 0
            last_time = time.time()
            while True:
                while time.time() - last_time < 0.008:  # 125 Hz
                    time.sleep(0.0001)
                last_time = time.time()

                if self.state == 'idle':
                    drift = 0
                elif self.state == 'moving':
                    drift += 0.0005
                    next_waypoint = self.waypoints[self.waypoint_index]
                    dist = distance(self.pose_odom, next_waypoint)
                    if dist < 0.01:
                        self.waypoint_index += 1
                        if self.waypoint_index == len(self.waypoints):
                            self.state = 'idle'
                    else:
                        x = self.pose_odom[0] + 0.01 * (next_waypoint[0] - self.pose_odom[0]) / dist
                        y = self.pose_odom[1] + 0.01 * (next_waypoint[1] - self.pose_odom[1]) / dist
                        self.pose_odom = (x, y, 0)
                        self.pose_map = (x + drift, y + drift, 0)
        finally:
            self.running = False

class ArmController:
    def __init__(self, robot_idx):
        self.robot_idx = robot_idx

        # Assumes parallel locking shafts are installed
        self.gripper_open_threshold = 2.5 / 229  # Fully open is 2/229
        self.gripper_closed_threshold = 226.5 / 229  # Fully closed is 227/229 (try restarting the arm if measured value is not 227/229)

        # Arm setup
        self.arm = KinovaArm()
        self.arm.clear_faults()
        self.arm.set_high_level_servoing()
        self.arm.set_joint_limits(4 * [80] + 3 * [70], [297] + 6 * [150])
        self.arm.set_max_twist_linear_limit()
        self.arm.move_angular([self.arm.get_heading(), 340, 180, 214, 0, 320, 90])
        if self.gripper_open_threshold < self.arm.get_gripper_position() < self.gripper_closed_threshold:
            gripper_state = 'closed'
        else:
            self.arm.open_gripper()
            gripper_state = 'open'
        print(f'Initial gripper state: {gripper_state}')

        # Control loop
        self.state = 'idle'  # States: idle, manipulating
        self.gripper_state = gripper_state
        self.target_ee_pos = None
        self.arm_heading = self.arm.get_heading()
        self.queue = Queue()

    def disconnect(self):
        self.arm.disconnect()

    def execute_command(self, command):
        self.queue.put(command)

    def stop(self):
        self.arm.apply_emergency_stop()
        time.sleep(0.05)  # Two 40 Hz cycles
        self.arm.clear_faults()
        self.arm.open_gripper()
        self.gripper_state = 'open'
        self.state = 'idle'

    def get_controller_data(self):
        return {
            'state': self.state,
            'gripper_state': self.gripper_state,
            'target_ee_pos': self.target_ee_pos,
            'arm_heading': self.arm_heading,
        }

    def check_grasp_success(self):
        gripper_position = self.arm.get_gripper_position()
        if gripper_position < self.gripper_closed_threshold:
            print(f'Grasp succeeded (gripper position: {gripper_position:.4f})')
            return True
        print(f'Grasp failed (gripper position: {gripper_position:.4f})')
        return False

    def pick_object(self, target_arm_heading, distance_to_target, grasp_orientation):
        self.arm.wait_ready()
        computed_joint_angles_1 = self.arm.compute_inverse_kinematics((distance_to_target, 0, -0.288 + 0.1), (180, 0, 90), guess_joint_angles=[0, 90, 180, 295, 0, 335, 90])
        computed_joint_angles_2 = self.arm.compute_inverse_kinematics((distance_to_target, 0, -0.288), (180, 0, 90), guess_joint_angles=[0, 90, 180, 295, 0, 335, 90])
        if computed_joint_angles_1 is not None and computed_joint_angles_2 is not None:
            computed_joint_angles_1[0] = target_arm_heading
            computed_joint_angles_2[0] = target_arm_heading
            computed_joint_angles_1[6] = grasp_orientation
            computed_joint_angles_2[6] = grasp_orientation
            self.arm.move_angular(computed_joint_angles_1)
            self.arm.move_angular(computed_joint_angles_2)
            self.arm.close_gripper()
            if self.check_grasp_success():
                self.gripper_state = 'closed'
                self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90], blocking=False)  # 2.03 secs
                self.arm.close_gripper(blocking=False)  # Make the grasp more secure
                time.sleep(1)  # Wait until arm reaches safe position before moving the base
            else:
                self.arm.open_gripper()
                self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90])

    def _check_gripper(self):
        # Check whether object slipped out of the gripper
        assert self.gripper_state == 'closed'
        self.arm.close_gripper()
        if not self.check_grasp_success():
            self.arm.open_gripper()
            self.gripper_state = 'open'
            return False
        return True

    def place_object(self, target_arm_heading, distance_to_target, height=-0.288, horizontal=False):  # height: -0.288 floor, 0.45 max
        if not self._check_gripper():
            return
        self.arm.wait_ready()
        theta_xyz = (90, 0, 90) if horizontal else (180, 0, 90)
        computed_joint_angles = self.arm.compute_inverse_kinematics((distance_to_target, 0, height), theta_xyz, guess_joint_angles=[0, 55, 180, 275, 0, 320, 90])
        if computed_joint_angles is not None:
            computed_joint_angles[0] = target_arm_heading
            self.arm.move_angular(computed_joint_angles)
            self.arm.open_gripper()
            self.gripper_state = 'open'
            self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90])

    def toss_object(self, target_arm_heading):
        if not self._check_gripper():
            return
        self.arm.wait_ready()
        self.arm.move_angular([target_arm_heading, 50, 180, 250, 0, 260, 90])
        self.arm.toss(target_arm_heading)
        self.gripper_state = 'open'
        self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90])

    def place_object_in_shelf(self, target_arm_heading, distance_to_target):
        if not self._check_gripper():
            return
        self.arm.wait_ready()
        computed_joint_angles_1 = self.arm.compute_inverse_kinematics((distance_to_target, 0, 0.05), (135, 0, 90), guess_joint_angles=[0, 60, 180, 275, 0, 5, 90])
        computed_joint_angles_2 = self.arm.compute_inverse_kinematics((distance_to_target + 0.15, 0, 0.05), (90, 0, 90), guess_joint_angles=[0, 90, 180, 305, 0, 50, 90])
        if computed_joint_angles_1 is not None and computed_joint_angles_2 is not None:
            computed_joint_angles_1[0] = target_arm_heading
            computed_joint_angles_2[0] = target_arm_heading
            self.arm.move_angular(computed_joint_angles_1)
            self.arm.move_angular(computed_joint_angles_2)
            self.arm.open_gripper()
            self.gripper_state = 'open'
            self.arm.move_angular(computed_joint_angles_1)
            self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90])

    def place_object_in_drawer(self, target_arm_heading, distance_to_target):
        if not self._check_gripper():
            return
        self.arm.wait_ready()
        computed_joint_angles_1 = self.arm.compute_inverse_kinematics((0.4, 0, -0.288), (180, 0, 90), guess_joint_angles=[0, 85, 180, 280, 0, 345, 90])
        computed_joint_angles_2 = self.arm.compute_inverse_kinematics((distance_to_target - 0.045, 0, 0.185), (90, 0, 90), guess_joint_angles=[0, 65, 180, 265, 0, 65, 90])
        computed_joint_angles_3 = self.arm.compute_inverse_kinematics((0.4, 0, -0.1), (150, 0, 90), guess_joint_angles=[0, 70, 180, 225, 0, 55, 90])
        computed_joint_angles_4 = self.arm.compute_inverse_kinematics((0.4, 0, 0.3), (180, 0, 90), guess_joint_angles=[0, 355, 180, 240, 0, 290, 90])
        computed_joint_angles_5 = self.arm.compute_inverse_kinematics((distance_to_target - 0.11, 0, 0.40), (160, 0, 90), guess_joint_angles=[0, 45, 180, 330, 0, 275, 90])
        computed_joint_angles_6 = self.arm.compute_inverse_kinematics((distance_to_target - 0.25, 0, 0.25), (160, 0, 90), guess_joint_angles=[0, 20, 180, 260, 0, 320, 90])
        if (computed_joint_angles_1 is not None and computed_joint_angles_2 is not None and computed_joint_angles_3 is not None
                and computed_joint_angles_4 is not None and computed_joint_angles_5 is not None and computed_joint_angles_6 is not None):
            computed_joint_angles_1[0] = target_arm_heading
            computed_joint_angles_2[0] = target_arm_heading
            computed_joint_angles_3[0] = target_arm_heading
            computed_joint_angles_4[0] = target_arm_heading
            computed_joint_angles_5[0] = target_arm_heading
            computed_joint_angles_6[0] = target_arm_heading

            # Set down object
            self.arm.move_angular(computed_joint_angles_1)
            self.arm.open_gripper()

            # Open drawer
            self.arm.move_angular(computed_joint_angles_2)
            self.arm.close_gripper()
            self.arm.move_cartesian((distance_to_target - 0.22, 0, 0.185), (90, 0, 90))
            self.arm.open_gripper()

            # Pick up object
            self.arm.move_angular(computed_joint_angles_3)
            self.arm.move_angular(computed_joint_angles_1)
            self.arm.close_gripper()
            self.arm.move_angular(computed_joint_angles_4)

            # Place object in drawer
            self.arm.move_angular(computed_joint_angles_5)
            self.arm.open_gripper()
            self.arm.move_angular(computed_joint_angles_6)

            # Close drawer
            self.arm.move_cartesian((distance_to_target - 0.04, 0, 0.185), (90, 0, 90))

            # Retract arm
            self.gripper_state = 'open'
            self.arm.move_angular([target_arm_heading, 340, 180, 214, 0, 320, 90])

    def run(self):
        while True:
            if self.queue.empty():
                self.state = 'idle'
            else:
                self.state = 'manipulating'
                try:
                    # Clear faults
                    if self.arm.in_fault():
                        self.arm.clear_faults()

                    arm_command = self.queue.get()

                    # Arm-dependent heading compensation
                    arm_command['target_arm_heading'] += ARM_HEADING_COMPENSATION[self.robot_idx]

                    # Execute command
                    self.target_ee_pos = arm_command['target_ee_pos']
                    if arm_command['primitive_name'] == 'pick':
                        self.pick_object(arm_command['target_arm_heading'], arm_command['distance_to_target'], arm_command['grasp_orientation'])
                    elif arm_command['primitive_name'] == 'place':
                        if arm_command['place_height'] is not None:
                            self.place_object(arm_command['target_arm_heading'], arm_command['distance_to_target'], height=arm_command['place_height'])
                        else:
                            self.place_object(arm_command['target_arm_heading'], arm_command['distance_to_target'])
                    elif arm_command['primitive_name'] == 'toss':
                        self.toss_object(arm_command['target_arm_heading'])
                    elif arm_command['primitive_name'] == 'shelf':
                        self.place_object_in_shelf(arm_command['target_arm_heading'], arm_command['distance_to_target'])
                    elif arm_command['primitive_name'] == 'drawer':
                        self.place_object_in_drawer(arm_command['target_arm_heading'], arm_command['distance_to_target'])

                    # Update state
                    self.target_ee_pos = None
                    self.arm_heading = self.arm.get_heading()
                except KServerException:
                    pass
            time.sleep(0.001)

class DummyArmController:
    def __init__(self, _):
        self.state = 'idle'  # States: idle, manipulating
        self.gripper_state = 'open'
        self.target_ee_pos = None
        self.arm_heading = 0
        self.queue = Queue()

    def disconnect(self):
        pass

    def execute_command(self, command):
        self.queue.put(command)

    def stop(self):
        pass

    def get_controller_data(self):
        return {
            'state': self.state,
            'gripper_state': self.gripper_state,
            'target_ee_pos': self.target_ee_pos,
            'arm_heading': self.arm_heading,
        }

    def run(self):
        while True:
            if self.queue.empty():
                self.state = 'idle'
            else:
                self.state = 'manipulating'
                arm_command = self.queue.get()
                self.target_ee_pos = arm_command['target_ee_pos']
                time.sleep(1)
                if arm_command['primitive_name'] == 'pick':
                    self.gripper_state = 'closed'
                else:
                    self.gripper_state = 'open'
                self.target_ee_pos = None
            time.sleep(0.001)

class Controller:
    def __init__(self, debug=False):
        self.state = 'idle'  # States: idle, moving, manipulating
        self.start_time = time.time()

        # Base controller
        try:
            self.base_controller = BaseController(debug=debug)
            robot_idx = self.base_controller.robot_idx
        except Exception as e:
            print(e)

            # Start dummy base controller
            self.base_controller = DummyBaseController()
            robot_idx = 0
            print('Could not start base controller, falling back to dummy base controller')

        # Arm controller
        try:
            self.arm_controller = ArmController(robot_idx)
        except Exception as e:
            print(e)

            # Start dummy arm controller
            self.arm_controller = DummyArmController(robot_idx)
            print('Could not start arm controller, falling back to dummy arm controller')

        # Object detection
        self.categories = []
        try:
            self.camera = Camera(robot_idx)
            self.object_detector_conn = Client((SERVER_HOSTNAME, 6003), authkey=CONN_AUTHKEY)
            self.queue = Queue()
            self.object_detector_conn.send({'encoded_image': self.camera.get_encoded_image(), 'categories': self.categories})  # Warm up the server
            self.object_detector_conn.recv()
            self.detected_object = None
        except Exception as e:
            print(e)
            self.camera = None
            print('Could not set up object detection, falling back to random selection from object categories')

        # Controller server
        self.server_conn = Client((SERVER_HOSTNAME, 6007 + robot_idx), authkey=CONN_AUTHKEY)

        # Logging
        log_dir = 'logs'
        Path(log_dir).mkdir(exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        logging.basicConfig(filename=f'{log_dir}/{today}.log', format='%(asctime)s %(levelname)s %(filename)s: %(message)s', level=logging.INFO, encoding='utf-8')
        logging.info('Starting new session')

    def _get_end_effector_offset(self, primitive_name):
        if self.arm_controller.gripper_state == 'open':
            return 0.55
        return {'toss': 1.30, 'shelf': 0.75, 'drawer': 0.80}.get(primitive_name, 0.55)

    def build_base_command(self, command):
        assert command['primitive_name'] in {'move', 'pick', 'place', 'toss', 'shelf', 'drawer'}

        # Base movement only
        if command['primitive_name'] == 'move':
            return {'waypoints': command['waypoints'], 'target_ee_pos': None, 'position_tolerance': 0.1}  # 10 cm tolerance for position error

        # Modify waypoints so that the end effector is placed at the target end effector position (the last waypoint)
        target_ee_pos = command['waypoints'][-1]
        end_effector_offset = self._get_end_effector_offset(command['primitive_name'])
        new_waypoint = None  # Find new_waypoint such that distance(new_waypoint, target_ee_pos) == end_effector_offset
        reversed_waypoints = command['waypoints'][::-1]
        for idx in range(1, len(reversed_waypoints)):
            start = reversed_waypoints[idx - 1]
            end = reversed_waypoints[idx]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - target_ee_pos[0], start[1] - target_ee_pos[1])
            t2 = intersect(d, f, end_effector_offset)
            if t2 is not None:
                new_waypoint = (start[0] + t2 * d[0], start[1] + t2 * d[1])
                break
        if new_waypoint is not None:
            # Discard all waypoints that are too close to target_ee_pos (distance(waypoint, target_ee_pos) < end_effector_offset)
            waypoints = reversed_waypoints[idx:][::-1] + [new_waypoint]
        else:
            # Base is too close to target end effector position and needs to back up
            print('Warning: Base needs to deviate from commanded path to reach target position, watch out for potential collisions')
            curr_position = command['waypoints'][0]
            signed_dist = distance(curr_position, target_ee_pos) - end_effector_offset
            dx = target_ee_pos[0] - curr_position[0]
            dy = target_ee_pos[1] - curr_position[1]
            target_heading = restrict_heading_range(math.atan2(dy, dx))
            target_position = (curr_position[0] + signed_dist * math.cos(target_heading), curr_position[1] + signed_dist * math.sin(target_heading))
            waypoints = [curr_position, target_position]
        base_command = {'waypoints': waypoints, 'target_ee_pos': target_ee_pos}

        return base_command

    def build_arm_command(self, command, base_pose):
        if command['primitive_name'] == 'move':
            return None
        target_ee_pos = command['waypoints'][-1]
        distance_to_target = distance(base_pose, target_ee_pos)
        end_effector_offset = self._get_end_effector_offset(command['primitive_name'])
        diff = abs(end_effector_offset - distance_to_target)
        if diff < 0.1:  # 10 cm
            dx = target_ee_pos[0] - base_pose[0]
            dy = target_ee_pos[1] - base_pose[1]
            target_arm_heading = math.degrees(math.pi - (math.atan2(dy, dx) - base_pose[2])) % 360
            grasp_orientation = 90
            if command.get('grasp_orientation', None) is not None:
                grasp_orientation += math.degrees(((command['grasp_orientation'] - base_pose[2]) + math.pi / 2) % math.pi - math.pi / 2)
                assert 0 <= grasp_orientation <= 180, grasp_orientation
            arm_command = {
                'primitive_name': command['primitive_name'],
                'target_ee_pos': target_ee_pos,
                'distance_to_target': distance_to_target,
                'target_arm_heading': target_arm_heading,
                'grasp_orientation': grasp_orientation,
                'place_height': command.get('place_height', None),
            }
            if arm_command['primitive_name'] == 'pick':
                assert self.arm_controller.gripper_state == 'open'
            return arm_command
        print(f'Too far from target end effector position ({(100 * diff):.1f} cm)')
        return None

    def handle_object_detection(self):
        while True:
            if self.state == 'moving':
                self.detected_object = None
            elif not self.queue.empty():
                self.queue.get()
                encoded_image = self.camera.get_encoded_image()  # 90 ms
                #self.object_detector_conn.send({'encoded_image': encoded_image, 'categories': self.categories})
                self.object_detector_conn.send({'encoded_image': encoded_image, 'categories': self.categories, 'use_clip': True})
                output = self.object_detector_conn.recv()  # 1000 ms
                logging.info(f'Object detection: {output["image_path"]} {list(zip(output["categories"], output["scores"]))}')  # pylint: disable=logging-fstring-interpolation
                self.detected_object = output['categories'][0]
            time.sleep(0.001)

    def request_object_detection(self):
        self.queue.put(None)

    def run(self):
        # Start base and arm controllers
        threading.Thread(target=self.base_controller.run, daemon=True).start()
        threading.Thread(target=self.arm_controller.run, daemon=True).start()

        # Start thread to handle object detection
        if self.camera is not None:
            threading.Thread(target=self.handle_object_detection, daemon=True).start()

        self.server_conn.send({'state': self.state, 'start_time': self.start_time})
        curr_command = None
        last_time = time.time()
        while self.base_controller.running:
            while time.time() - last_time < 0.008:  # 125 Hz
                time.sleep(0.0001)
            step_time = time.time() - last_time
            if step_time > 0.02:  # 50 Hz
                print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__}')
            last_time = time.time()

            # Communicate with server
            new_command = None
            if self.server_conn.poll():
                new_command = self.server_conn.recv()
                controller_data = {'state': self.state, 'start_time': self.start_time}
                if self.camera is not None:
                    controller_data['detected_object'] = self.detected_object
                elif len(self.categories) > 0:
                    controller_data['detected_object'] = random.choice(self.categories)
                controller_data['base'] = self.base_controller.get_controller_data()
                controller_data['arm'] = self.arm_controller.get_controller_data()
                self.server_conn.send(controller_data)

            # Base and arm control logic
            if new_command == 'stop':
                if self.state == 'moving':
                    self.base_controller.stop()
                elif self.state == 'manipulating':
                    self.arm_controller.stop()
                self.state = 'idle'
            elif self.state == 'idle':
                if new_command is not None:  # Only accept new commands if robot is idle
                    curr_command = new_command
                    logging.info(f'Executing command: {curr_command}')  # pylint: disable=logging-fstring-interpolation
                    self.categories = curr_command.get('categories', [])
                    self.state = 'moving'
                    self.base_controller.execute_command(self.build_base_command(curr_command))
            elif self.state == 'moving':
                if self.base_controller.state == 'idle':
                    arm_command = self.build_arm_command(curr_command, self.base_controller.pose_map)
                    if arm_command is not None:
                        if self.camera is not None and arm_command['primitive_name'] == 'pick':
                            self.request_object_detection()
                        self.state = 'manipulating'
                        self.arm_controller.execute_command(arm_command)
                    else:
                        self.state = 'idle'
            elif self.state == 'manipulating':
                if self.arm_controller.state == 'idle':
                    self.state = 'idle'

        print('Disconnecting arm')
        self.arm_controller.disconnect()
        if self.camera is not None:
            print('Disconnecting camera')
            self.camera.disconnect()
        time.sleep(1)  # Wait for error messages in threads to print

def main(args):
    Controller(debug=args.debug).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
