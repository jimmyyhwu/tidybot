# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from multiprocessing import Process
import cv2 as cv
import utils
from camera_server import CameraServer
from controller_client import ControllerClient
from controller_server import ControllerServer
from image_client import ImageClient
from marker_detector_server import MarkerDetectorServer
from occupancy_map import OccupancyMap

class Teleop:
    def __init__(self, robot_idx, scale_factor=0.35):
        self.robot_idx = robot_idx
        self.scale_factor = scale_factor
        self.robot_state = 'idle'  # States: idle, moving
        self.robot_position = (0, 0)
        self.robot_gripper_state = None
        self.waypoints = [self.robot_position]

        # Graphical interface
        self.image_client = ImageClient(port1=6000, port2=6001, scale_factor=self.scale_factor)
        self.image_shape = self.image_client.image_shape
        self.window_name = f'Teleop (Robot {self.robot_idx + 1})'
        cv.namedWindow(self.window_name)
        cv.setMouseCallback(self.window_name, self.on_mouse)

    def on_mouse(self, event, x, y, *_):
        if self.robot_state == 'idle' and event == cv.EVENT_LBUTTONDOWN:
            target_position = self.pixel_xy_to_position((x, y))
            self.waypoints.append(target_position)

    def reset_waypoints(self):
        self.waypoints = [self.robot_position]

    def position_to_pixel_xy(self, position):
        return utils.position_to_pixel_xy(position, self.image_shape, self.scale_factor)

    def pixel_xy_to_position(self, pixel_xy):
        return utils.pixel_xy_to_position(pixel_xy, self.image_shape, self.scale_factor)

    def store_controller_data(self, controller_data):
        if 'base' in controller_data:
            self.robot_position = (controller_data['base']['pose'][0], controller_data['base']['pose'][1])
        if 'arm' in controller_data:
            self.robot_gripper_state = controller_data['arm']['gripper_state']

    def run(self):
        # Connect to controller
        controller = ControllerClient(self.robot_idx)

        # Set up the different primitives
        default_idx = 0
        primitive_names = ['move', 'place', 'drop', 'toss', 'shelf', 'drawer']
        primitive_name = primitive_names[default_idx]
        primitive_descs = [
            'Movement-only mode',
            'Pick and place mode',
            'Pick and drop mode',
            'Pick and toss mode',
            'Pick and shelf mode',
            'Pick and drawer mode',
        ]
        print('Place primitives:')
        for i, desc in enumerate(primitive_descs):
            if i == default_idx:
                print(f'{i}. {desc} (default)')
            else:
                print(f'{i}. {desc}')

        # Run GUI loop
        idle_steps = 0
        esc_pressed_count = 0
        last_time = time.time()
        while True:
            step_time = time.time() - last_time
            if step_time > 0.07:  # 14 Hz
                print(f'Warning: Step time {1000 * step_time:.1f} ms in {self.__class__.__name__}')
            last_time = time.time()

            # Update robot state
            controller_data = controller.get_controller_data()  # 30 ms
            if controller_data['state'] == 'moving':
                self.robot_state = 'moving'
            elif controller_data['state'] == 'idle' and self.robot_state == 'moving':
                idle_steps += 1
            if idle_steps > 3:
                idle_steps = 0
                self.robot_state = 'idle'
                self.reset_waypoints()

            # Store controller data
            self.store_controller_data(controller_data)
            if self.robot_state == 'idle':
                self.waypoints[0] = self.robot_position

            # Check for keypress
            key = cv.waitKey(1)
            if key == ord('q') or cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 0.5:
                break
            if key == 27:  # <Esc>
                controller.stop()
                self.reset_waypoints()
                esc_pressed_count += 1
                if esc_pressed_count > 1:
                    controller.reset_visualizer()
            elif key == 13 and len(self.waypoints) > 1:  # <Enter>
                command = {'primitive_name': primitive_name, 'waypoints': self.waypoints}
                if primitive_name != 'move' and self.robot_gripper_state == 'open':
                    command['primitive_name'] = 'pick'
                elif primitive_name == 'drop':
                    command['primitive_name'] = 'place'
                    command['place_height'] = 0.3
                controller.execute_command(command)
                self.reset_waypoints()
                self.robot_state = 'moving'
                esc_pressed_count = 0
            elif ord(str(0)) <= key < ord(str(len(primitive_names))):
                idx = key - ord(str(0))
                primitive_name = primitive_names[idx]
                print(primitive_descs[idx])

            # Draw waypoints
            image = self.image_client.get_image()
            if len(self.waypoints) > 1:
                waypoints = list(map(self.position_to_pixel_xy, self.waypoints))
                for i in range(1, len(waypoints)):
                    cv.line(image, waypoints[i - 1], waypoints[i], (255, 0, 0), 2)
            cv.imshow(self.window_name, image)

        cv.destroyAllWindows()

class TeleopShortestPath(Teleop):
    def __init__(self, robot_idx, scale_factor=0.35, debug=False):
        super().__init__(robot_idx, scale_factor=scale_factor)
        scenario = utils.load_yaml('scenarios/test.yml')
        self.occupancy_map = OccupancyMap(obstacles=scenario['receptacles'].values(), debug=debug)

    def on_mouse(self, event, x, y, *_):
        if self.robot_state == 'idle' and event == cv.EVENT_LBUTTONDOWN:
            target_position = self.pixel_xy_to_position((x, y))
            self.waypoints = self.occupancy_map.shortest_path(self.robot_position, target_position)

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
    def start_controller_server():
        ControllerServer(robot_idx, debug=args.debug).run()
    Process(target=start_controller_server, daemon=True).start()

    # Start teleop
    if args.shortest_path:
        TeleopShortestPath(robot_idx, debug=args.debug).run()
    else:
        Teleop(robot_idx).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-num', type=int, default=1)
    parser.add_argument('--shortest-path', action='store_true')
    parser.add_argument('--debug', action='store_true')
    main(parser.parse_args())
