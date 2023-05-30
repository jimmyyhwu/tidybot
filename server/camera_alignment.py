# Author: Jimmy Wu
# Date: February 2023

import argparse
import json
from pathlib import Path
import cv2 as cv
import numpy as np
import utils

def compute_transformation_matrix(src_points):
    dst_points = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    return cv.getPerspectiveTransform(np.array(src_points, dtype=np.float32), dst_points).astype(np.float32)

class Annotator:
    def __init__(self, serial, center=False):
        self.serial = serial
        self.center = center
        self.window_name = 'window'
        cv.namedWindow(self.window_name)
        self.image_width, self.image_height, self.camera_matrix, self.dist_coeffs = utils.get_camera_params(self.serial)
        self.save_path = Path('camera_params') / f'{self.serial}.json'
        self.labels = self.load_labels()

        # Camera center
        self.image_center = (self.image_width // 2, self.image_height // 2)
        if 'camera_center' in self.labels:
            self.camera_center = self.labels['camera_center']
        else:
            self.camera_center = None
        if 'camera_center_relative' in self.labels:  # Relative position
            self.camera_center_relative = self.labels['camera_center_relative']
        else:
            self.camera_center_relative = (0.5, 0.5)

        # Camera corners
        if 'camera_corners' in self.labels:
            self.camera_corners = self.labels['camera_corners']
        else:
            default_padding = 50
            self.camera_corners = [
                (default_padding, default_padding),
                (self.image_width - default_padding, default_padding),
                (self.image_width - default_padding, self.image_height - default_padding),
                (default_padding, self.image_height - default_padding),
            ]

        cv.setMouseCallback(self.window_name, self.on_mouse)

    def load_labels(self):
        if not self.save_path.exists():
            return {}
        with open(self.save_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_labels(self):
        if not self.save_path.parent.exists():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f)
        print(self.labels)
        print(f'Saved to {self.save_path}')

    def on_mouse(self, event, x, y, *_):
        if event == cv.EVENT_LBUTTONDBLCLK:
            if self.center:
                self.camera_center = (x, y)
                transformation_matrix = compute_transformation_matrix(self.camera_corners)
                camera_center = np.array([[self.camera_center]], dtype=np.float32)
                self.camera_center_relative = cv.perspectiveTransform(camera_center, transformation_matrix).reshape(2).tolist()
            else:
                idx = np.argmin([utils.distance(pt, (x, y)) for pt in self.camera_corners])
                self.camera_corners[idx] = (x, y)
                _, transformation_matrix = cv.invert(compute_transformation_matrix(self.camera_corners))
                camera_center_relative = np.array([[self.camera_center_relative]], dtype=np.float32)
                self.camera_center = cv.perspectiveTransform(camera_center_relative, transformation_matrix).reshape(2).astype(int).tolist()
            self.labels['camera_corners'] = self.camera_corners
            self.labels['camera_center'] = self.camera_center
            self.labels['camera_center_relative'] = self.camera_center_relative
            self.save_labels()

    def process_image(self, image):
        # Camera center
        if self.center:
            cv.circle(image, self.image_center, 20, (0, 255, 0))
        if self.camera_center is not None:
            cv.circle(image, self.camera_center, 5, (0, 0, 255))

        # Camera corners
        for i, corner in enumerate(self.camera_corners):
            cv.circle(image, corner, 5, (0, 0, 255))
            cv.line(image, corner, self.camera_corners[(i + 1) % 4], (0, 0, 255))

    def run(self):
        cap = utils.get_video_cap(self.serial, self.image_width, self.image_height)
        while True:
            # Brighten image
            cap.set(cv.CAP_PROP_EXPOSURE, 624)
            cap.set(cv.CAP_PROP_GAIN, 0)

            if cv.waitKey(1) == 27 or cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 0.5:
                break

            image = None
            while image is None:
                _, image = cap.read()

            image = cv.undistort(image, self.camera_matrix, self.dist_coeffs)
            self.process_image(image)
            cv.imshow(self.window_name, image)

        cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=None)
    parser.add_argument('--center', action='store_true')
    parser.add_argument('--camera2', action='store_true')  # Use bottom camera
    args = parser.parse_args()
    if args.serial is None:
        if args.camera2:
            args.serial = '099A11EE'
        else:
            args.serial = 'E4298F4E'
    Annotator(args.serial, center=args.center).run()
