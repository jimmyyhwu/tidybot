# Author: Jimmy Wu
# Date: February 2023

import argparse
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process
from multiprocessing.connection import Client
from pathlib import Path
import cv2 as cv
import numpy as np
import utils
from camera_server import CameraServer
from constants import CAMERA_SERIALS, CAMERA_HEIGHT
from constants import CONN_AUTHKEY
from constants import FLOOR_LENGTH, FLOOR_WIDTH
from constants import ROBOT_WIDTH, ROBOT_HEIGHT
from controller_client import ControllerClient
from controller_server import ControllerServer
from image_client import ImageClient
from marker_detector_server import MarkerDetectorServer
from object_detector_server import ObjectDetectorServer, ObjectDetectorVisualizer
from occupancy_map import OccupancyMap

def get_grasp(mask):
    # Grasp center
    dist = cv.distanceTransform(mask, cv.DIST_L2, 5)  # From https://stackoverflow.com/a/65409262
    center = tuple(map(round, np.argwhere(dist == dist.max()).mean(axis=0)))
    center = (center[1], center[0]) # (i, j) to (x, y)

    # Grasp orientation
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv.contourArea)
        moments = cv.moments(largest_contour)
        orientation = 0.5 * math.atan2(2 * moments['mu11'], moments['mu20'] - moments['mu02'])  # From https://en.wikipedia.org/wiki/Image_moment
        orientation = -orientation  # y goes up but j goes down
    else:
        orientation = None

    return center, orientation

class Demo:
    def __init__(self, robot_idx, scenario_name, scale_factor=0.35, debug=False):
        self.robot_idx = robot_idx
        self.scale_factor = scale_factor
        self.debug = debug
        self.robot_state = 'idle'  # States: idle, moving
        self.robot_pose = (0, 0, 0)
        self.robot_position = (0, 0)
        self.robot_gripper_state = None
        self.robot_arm_heading = None
        self.detected_object = None
        self.command = None
        self.last_command_time = 0
        self.num_objs_per_recep = defaultdict(int)

        # Set up scenario
        scenario = utils.load_yaml(f'scenarios/{scenario_name}.yml')
        self.receptacles = scenario['receptacles']
        self.occupancy_map = OccupancyMap(obstacles=self.receptacles.values(), debug=self.debug)
        self.preferences = utils.load_yaml(f'preferences/{scenario_name}.yml')

        # Image client for overhead images
        self.image_client = ImageClient(port1=6000, port2=6001, scale_factor=self.scale_factor)
        self.image_shape = self.image_client.image_shape

        # Graphical interface
        self.window_name = f'Demo (Robot {self.robot_idx + 1})'
        cv.namedWindow(self.window_name)

        # Logging
        log_dir = 'logs'
        Path(log_dir).mkdir(exist_ok=True)
        today = datetime.now().strftime('%Y-%m-%d')
        logging.basicConfig(filename=f'{log_dir}/{today}.log', format='%(asctime)s %(levelname)s %(filename)s: %(message)s', level=logging.INFO, encoding='utf-8')
        logging.info('Starting new session')

    def position_to_pixel_xy(self, position):
        return utils.position_to_pixel_xy(position, self.image_shape, self.scale_factor)

    def pixel_xy_to_position(self, pixel_xy):
        return utils.pixel_xy_to_position(pixel_xy, self.image_shape, self.scale_factor)

    def filter_detection_output(self, output):
        if len(output['boxes']) == 0:
            return output

        camera_center_offset_x = FLOOR_WIDTH * ((0.5486 + 0.5506) / 2 - 0.5)  # 18 cm (from camera_params/*.json)
        ignore_boxes = []

        # Base at floor height (assumes heading 0)
        min_xy = self.position_to_pixel_xy((self.robot_position[0] - ROBOT_WIDTH / 2, self.robot_position[1] + ROBOT_WIDTH / 2))
        max_xy = self.position_to_pixel_xy((self.robot_position[0] + ROBOT_WIDTH / 2, self.robot_position[1] - ROBOT_WIDTH / 2))
        ignore_boxes.append([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

        # Top of base (assumes heading 0)
        height_ratio_base = CAMERA_HEIGHT / (CAMERA_HEIGHT - ROBOT_HEIGHT)
        if self.robot_position[1] < 0:
            robot_center_top = (
                camera_center_offset_x + height_ratio_base * (self.robot_position[0] - camera_center_offset_x),
                (-FLOOR_LENGTH) / 4 + height_ratio_base * (self.robot_position[1] + FLOOR_LENGTH / 4))
        else:
            robot_center_top = (
                camera_center_offset_x + height_ratio_base * (self.robot_position[0] - camera_center_offset_x),
                FLOOR_LENGTH / 4 + height_ratio_base * (self.robot_position[1] - FLOOR_LENGTH / 4))
        robot_width_top = height_ratio_base * ROBOT_WIDTH
        min_xy = self.position_to_pixel_xy((robot_center_top[0] - robot_width_top / 2, robot_center_top[1] + robot_width_top / 2))
        max_xy = self.position_to_pixel_xy((robot_center_top[0] + robot_width_top / 2, robot_center_top[1] - robot_width_top / 2))
        ignore_boxes.append([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

        # Arm
        if self.robot_arm_heading is not None:
            arm_width = 0.15

            # Joint 4 of 7
            height_ratio_arm_joint_4 = CAMERA_HEIGHT / (CAMERA_HEIGHT - (ROBOT_HEIGHT + 0.69))
            theta = self.robot_pose[2] - math.radians(self.robot_arm_heading)
            arm_position_joint_4 = (self.robot_pose[0] + 0.15 * math.cos(theta),
                                    self.robot_pose[1] + 0.15 * math.sin(theta))
            if arm_position_joint_4[1] < 0:
                arm_center_joint_4 = (
                    camera_center_offset_x + height_ratio_arm_joint_4 * (arm_position_joint_4[0] - camera_center_offset_x),
                    (-FLOOR_LENGTH) / 4 + height_ratio_arm_joint_4 * (arm_position_joint_4[1] + FLOOR_LENGTH / 4))
            else:
                arm_center_joint_4 = (
                    camera_center_offset_x + height_ratio_arm_joint_4 * (arm_position_joint_4[0] - camera_center_offset_x),
                    FLOOR_LENGTH / 4 + height_ratio_arm_joint_4 * (arm_position_joint_4[1] - FLOOR_LENGTH / 4))
            arm_width_joint_4 = height_ratio_arm_joint_4 * arm_width
            min_xy = self.position_to_pixel_xy((arm_center_joint_4[0] - arm_width_joint_4 / 2, arm_center_joint_4[1] + arm_width_joint_4 / 2))
            max_xy = self.position_to_pixel_xy((arm_center_joint_4[0] + arm_width_joint_4 / 2, arm_center_joint_4[1] - arm_width_joint_4 / 2))
            ignore_boxes.append([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

            # Joint 6 of 7
            height_ratio_arm_joint_6 = CAMERA_HEIGHT / (CAMERA_HEIGHT - (ROBOT_HEIGHT + 0.51))
            arm_position_joint_6 = (self.robot_pose[0] - 0.11 * math.cos(theta),
                                    self.robot_pose[1] - 0.11 * math.sin(theta))
            if arm_position_joint_6[1] < 0:
                arm_center_joint_6 = (
                    camera_center_offset_x + height_ratio_arm_joint_6 * (arm_position_joint_6[0] - camera_center_offset_x),
                    (-FLOOR_LENGTH) / 4 + height_ratio_arm_joint_6 * (arm_position_joint_6[1] + FLOOR_LENGTH / 4))
            else:
                arm_center_joint_6 = (
                    camera_center_offset_x + height_ratio_arm_joint_6 * (arm_position_joint_6[0] - camera_center_offset_x),
                    FLOOR_LENGTH / 4 + height_ratio_arm_joint_6 * (arm_position_joint_6[1] - FLOOR_LENGTH / 4))
            arm_width_joint_6 = height_ratio_arm_joint_6 * arm_width
            min_xy = self.position_to_pixel_xy((arm_center_joint_6[0] - arm_width_joint_6 / 2, arm_center_joint_6[1] + arm_width_joint_6 / 2))
            max_xy = self.position_to_pixel_xy((arm_center_joint_6[0] + arm_width_joint_6 / 2, arm_center_joint_6[1] - arm_width_joint_6 / 2))
            ignore_boxes.append([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

        # Receptacles
        # Note: Due to camera perspective distortion, receptacles should be placed near edges of overhead camera image frame
        for receptacle in self.receptacles.values():
            pos_x, pos_y = receptacle['position']
            dim_x, dim_y = receptacle['dimensions']
            min_xy = self.position_to_pixel_xy((pos_x - dim_x / 2, pos_y + dim_y / 2))
            max_xy = self.position_to_pixel_xy((pos_x + dim_x / 2, pos_y - dim_y / 2))
            ignore_boxes.append([min_xy[0], min_xy[1], max_xy[0], max_xy[1]])

        # output['boxes'] = ignore_boxes
        # output['masks'] = len(ignore_boxes) * output['masks'][0]
        # output['categories'] = ['robot_floor', 'robot_top', 'arm_joint_4', 'arm_joint_6', *self.receptacles.keys()]
        # output['scores'] = len(ignore_boxes) * [1.0]
        # return output

        keep_indices = []
        for i, box in enumerate(output['boxes']):
            skip = False

            # Skip detections that overlap with ignore boxes
            for ignore_box in ignore_boxes:
                w = max(0, min(box[2], ignore_box[2]) - max(box[0], ignore_box[0]))
                h = max(0, min(box[3], ignore_box[3]) - max(box[1], ignore_box[1]))
                intersection = w * h
                if intersection > 0:
                    skip = True
                    break

            if not skip:
                keep_indices.append(i)

        keep_indices = np.array(keep_indices, dtype=np.int64)  # Explicit dtype needs to be specified for empty list
        output['boxes'] = output['boxes'][keep_indices]
        output['masks'] = output['masks'][keep_indices]
        output['categories'] = [output['categories'][i] for i in keep_indices]
        output['scores'] = output['scores'][keep_indices]

        return output

    def distance(self, position):
        path = self.occupancy_map.shortest_path(self.robot_position, position)
        return sum(utils.distance(path[i - 1], path[i]) for i in range(1, len(path)))

    def get_closest_grasp(self, boxes, masks):
        best_dist = math.inf
        center = None
        orientation = None
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            box_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            dist = self.distance(self.pixel_xy_to_position(box_center))  # 30 ms
            if dist < best_dist:
                best_dist = dist
                center, orientation = get_grasp(masks[i])  # 4 ms
        return center, orientation

    def update_command(self, detection_output=None):
        if self.robot_state != 'idle':
            return

        if self.robot_gripper_state == 'open':  # Pick up object
            if detection_output is None:  # Wait for detection output
                return

            grasp_center, grasp_orientation = self.get_closest_grasp(detection_output['boxes'], detection_output['masks'])
            if grasp_center is not None:
                # Pick up closest object
                self.command = {
                    'primitive_name': 'pick',
                    'waypoints': self.occupancy_map.shortest_path(self.robot_position, self.pixel_xy_to_position(grasp_center)),
                    'grasp_orientation': grasp_orientation,
                    'categories': self.preferences['categories'],
                    'image_path': detection_output['image_path'],
                }
            else:
                # No objects found
                self.command = None

        elif self.robot_gripper_state == 'closed':
            if self.detected_object is not None:  # Bring object to receptacle
                # Select receptacle
                receptacle_name = self.preferences['placements'][self.detected_object]
                receptacle = self.receptacles[receptacle_name]
                target_ee_pos = (receptacle['position'][0], receptacle['position'][1])

                # Select primitive
                primitive_name = self.preferences['primitives'].get(self.detected_object, 'place')  # place or toss
                if primitive_name not in receptacle['primitive_names']:
                    primitive_name = 'place'
                if primitive_name == 'toss' and utils.distance(self.robot_position, target_ee_pos) < 0.9:  # If close to receptacle, just place instead of toss
                    primitive_name = 'place'
                if 'shelf' in receptacle_name:
                    primitive_name = 'shelf'
                elif 'drawer' in receptacle_name:
                    primitive_name = 'drawer'

                # Build command
                command = {'receptacle_name': receptacle_name, 'primitive_name': primitive_name}
                if 'center_offset' in receptacle:
                    target_ee_pos = (receptacle['position'][0] + receptacle['center_offset'][0],
                                     receptacle['position'][1] + receptacle['center_offset'][1])
                    if receptacle_name in {'coffee table', 'sofa'}:
                        # Avoid stacking objects on top of each other
                        num_objs = self.num_objs_per_recep[receptacle_name]
                        offset = ((num_objs + 1) // 2) * (1 if num_objs % 2 == 0 else -1)
                        if receptacle['dimensions'][0] > receptacle['dimensions'][1]:  # Longer in x-direction
                            target_ee_pos = (target_ee_pos[0] + 0.1 * offset * receptacle['dimensions'][0], target_ee_pos[1])
                        else:  # Longer in y-direction
                            target_ee_pos = (target_ee_pos[0], target_ee_pos[1] + 0.1 * offset * receptacle['dimensions'][1])
                if 'end_effector_offset' in receptacle:
                    if receptacle['dimensions'][0] > receptacle['dimensions'][1]:  # Longer in x-direction
                        new_waypoint = (target_ee_pos[0], target_ee_pos[1] + math.copysign(1, receptacle['center_offset'][1]) * receptacle['end_effector_offset'])
                    else:  # Longer in y-direction
                        new_waypoint = (target_ee_pos[0] + math.copysign(1, receptacle['center_offset'][0]) * receptacle['end_effector_offset'], target_ee_pos[1])
                    command['waypoints'] = self.occupancy_map.shortest_path(self.robot_position, new_waypoint) + [target_ee_pos]
                else:
                    command['waypoints'] = self.occupancy_map.shortest_path(self.robot_position, target_ee_pos)
                if primitive_name == 'place':
                    command['place_height'] = receptacle['place_height']
                self.command = command

            else:  # Object detection took too long, put object back down
                target_ee_pos = (self.robot_position[0] - 0.55 * math.cos(self.robot_pose[2]),
                                 self.robot_position[1] - 0.55 * math.sin(self.robot_pose[2]))
                self.command = {'primitive_name': 'place', 'waypoints': [self.robot_position, target_ee_pos]}

    def store_controller_data(self, controller_data):
        if 'base' in controller_data:
            self.robot_pose = controller_data['base']['pose']
            self.robot_position = (self.robot_pose[0], self.robot_pose[1])
        if 'arm' in controller_data:
            self.robot_gripper_state = controller_data['arm']['gripper_state']
            self.robot_arm_heading = controller_data['arm']['arm_heading']
        self.detected_object = controller_data.get('detected_object', None)

    def run(self):
        # Connect to controller
        controller = ControllerClient(self.robot_idx)

        # Set up object detection
        object_detector_conn = Client(('localhost', 6003), authkey=CONN_AUTHKEY)
        if self.debug:
            visualizer = ObjectDetectorVisualizer()

        # Set up the different operating modes
        default_idx = 0
        operating_modes = ['supervised', 'autonomous']
        operating_mode = operating_modes[default_idx]
        operating_mode_descs = [
            'Supervised mode',
            'Autonomous mode',
        ]
        print('Operating mode:')
        for i, desc in enumerate(operating_mode_descs):
            if i == default_idx:
                print(f'{i}. {desc} (default)')
            else:
                print(f'{i}. {desc}')

        # Run GUI loop
        idle_steps = 0
        esc_pressed_count = 0
        autonomous_running = False
        detection_image_sent = False
        last_time = time.time()
        controller_state_time = None
        while True:
            step_time = time.time() - last_time
            if step_time > 0.5:  # 2 Hz
                print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__}')
            last_time = time.time()

            # Update robot state
            controller_data = controller.get_controller_data()  # 30 ms
            if controller_data['start_time'] != controller_state_time:  # Controller was restarted
                controller_state_time = controller_data['start_time']
                autonomous_running = False
            if controller_data['state'] == 'moving':
                self.robot_state = 'moving'
            elif controller_data['state'] == 'idle' and self.robot_state == 'moving':
                idle_steps += 1
            if idle_steps > 3:
                idle_steps = 0
                self.robot_state = 'idle'
                self.command = None

            # Store controller data
            self.store_controller_data(controller_data)

            # Check for keypress
            key = cv.waitKey(1)
            if key == ord('q') or cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 0.5:
                break
            if key == 27:  # <Esc>
                controller.stop()
                esc_pressed_count += 1
                if esc_pressed_count > 1:
                    controller.reset_visualizer()
                autonomous_running = False
            elif key == 13:  # <Enter>
                esc_pressed_count = 0
                autonomous_running = True
            elif ord(str(0)) <= key < ord(str(len(operating_modes))):
                idx = key - ord(str(0))
                operating_mode = operating_modes[idx]
                print(operating_mode_descs[idx])

            # Send command
            if self.robot_state == 'idle' and autonomous_running and self.command is not None and time.time() - self.last_command_time > 1.0:
                logging.info(f'Sending command: {self.command}')  # pylint: disable=logging-fstring-interpolation
                controller.execute_command(self.command)
                self.last_command_time = time.time()  # Wait at least 1 sec between commands
                if 'receptacle_name' in self.command:
                    self.num_objs_per_recep[self.command['receptacle_name']] += 1
                self.robot_state = 'moving'
            if operating_mode == 'supervised':  # In supervised mode, every command is manually approved
                autonomous_running = False

            # Draw waypoints and grasp orientation
            image = self.image_client.get_image()
            image_copy = image.copy()
            if self.command is not None:
                waypoints = list(map(self.position_to_pixel_xy, self.command['waypoints']))
                for i in range(1, len(waypoints)):
                    cv.line(image_copy, waypoints[i - 1], waypoints[i], (255, 0, 0), 2)
                grasp_orientation = self.command.get('grasp_orientation', None)
                if grasp_orientation is not None:
                    grasp_center = self.command['waypoints'][-1]
                    robotiq_gripper_width = 0.085
                    grasp_line = [(grasp_center[0] - robotiq_gripper_width / 2 * math.sin(grasp_orientation),
                                   grasp_center[1] + robotiq_gripper_width / 2 * math.cos(grasp_orientation)),
                                  (grasp_center[0] + robotiq_gripper_width / 2 * math.sin(grasp_orientation),
                                   grasp_center[1] - robotiq_gripper_width / 2 * math.cos(grasp_orientation))]
                    cv.line(image_copy, self.position_to_pixel_xy(grasp_line[0]), self.position_to_pixel_xy(grasp_line[1]), (255, 0, 0), 2)
            cv.imshow(self.window_name, image_copy)

            # Object detection
            detection_output = None
            if object_detector_conn.poll():
                detection_output = object_detector_conn.recv()
                detection_image_sent = False
                detection_output = self.filter_detection_output(detection_output)
                if self.debug and self.robot_state == 'idle':
                    visualizer.visualize(cv.cvtColor(image, cv.COLOR_BGR2RGB), detection_output)
            if self.robot_state == 'idle' and not detection_image_sent:
                _, encoded_image = cv.imencode('.jpg', image)
                object_detector_conn.send({'encoded_image': encoded_image, 'categories': ['object'], 'min_box_area': 36, 'max_box_area': 40000})
                detection_image_sent = True

            # Update command
            self.update_command(detection_output=detection_output)

        cv.destroyAllWindows()

def main(args):
    # Start camera servers
    def start_camera_server(serial, port):
        CameraServer(serial, port=port).run()
    for serial, port in [(CAMERA_SERIALS[0], 6000), (CAMERA_SERIALS[1], 6001)]:
        Process(target=start_camera_server, args=(serial, port), daemon=True).start()

    # Wait for camera servers to be ready
    time.sleep(1.5)

    # Start marker detector server
    def start_marker_detector_server():
        MarkerDetectorServer(hostname='0.0.0.0').run()
    Process(target=start_marker_detector_server, daemon=True).start()

    # Start controller server
    robot_idx = args.robot_num - 1
    def start_controller_server():
        ControllerServer(robot_idx).run()
    Process(target=start_controller_server, daemon=True).start()

    # Start object detector server
    # def start_object_detector_server():
    #     ObjectDetectorServer(hostname='0.0.0.0').run()
    # Process(target=start_object_detector_server, daemon=True).start()

    # Run demo
    Demo(robot_idx, args.scenario_name, debug=args.debug).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-num', type=int, default=1)
    parser.add_argument('--scenario-name', default='test')
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
