# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from pathlib import Path
import cv2 as cv
import numpy as np
import utils
from camera_client import CameraClient
from constants import PIXELS_PER_M, FLOOR_LENGTH, FLOOR_WIDTH

def compute_transformation_matrix(src_points, image_width, image_height):
    dst_points = np.array([
        [0, 0],
        [image_width, 0],
        [image_width, image_height],
        [0, image_height],
    ], dtype=np.float32)
    return cv.getPerspectiveTransform(src_points, dst_points)

class CameraClientWrapper:
    def __init__(self, serial, port, scale_factor=1.0):
        self.image_width = round(scale_factor * PIXELS_PER_M * FLOOR_WIDTH)
        self.image_height = round(scale_factor * PIXELS_PER_M * FLOOR_LENGTH / 2)
        print(f'Image size: ({self.image_width}, {self.image_height})')
        _, camera_corners = utils.get_camera_alignment_params(serial)
        self.transformation_matrix = compute_transformation_matrix(np.array(camera_corners, dtype=np.float32), self.image_width, self.image_height)
        self.camera_client = CameraClient(port)

    def get_image(self):
        return cv.warpPerspective(self.camera_client.get_image(), self.transformation_matrix, (self.image_width, self.image_height))

class ImageClient:
    def __init__(self, top_only=False, port1=6000, port2=6001, scale_factor=1.0):
        if top_only:
            self.cameras = [CameraClientWrapper('E4298F4E', port1, scale_factor=scale_factor)]
        else:
            self.cameras = [CameraClientWrapper('E4298F4E', port1, scale_factor=scale_factor), CameraClientWrapper('099A11EE', port2, scale_factor=scale_factor)]
        image_width = self.cameras[0].image_width
        image_height = sum(camera.image_height for camera in self.cameras)
        self.image_shape = (image_height, image_width)

    def get_image(self):
        return np.concatenate([camera.get_image() for camera in self.cameras], axis=0)

    def close(self):
        for camera in self.cameras:
            camera.camera_client.close()

def main(args):
    image_client = ImageClient(top_only=args.top_only, port1=args.port1, port2=args.port2, scale_factor=args.scale_factor)
    window_name = 'out'

    try:
        while True:
            if args.benchmark:
                start_time = time.time()

            image = image_client.get_image()

            if not args.benchmark:
                cv.imshow(window_name, image)
                key = cv.waitKey(1)
                if key == 27 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 0.5:
                    break
                if key == ord('s'):  # Save image
                    image_dir = Path('images')
                    if not image_dir.exists():
                        image_dir.mkdir()
                    image_path = str(image_dir / f'image-{int(10 * time.time()) % 100000000}.jpg')
                    cv.imwrite(image_path, image)
                    print(f'Saved image to {image_path} (scale factor {args.scale_factor})')

            if args.benchmark:
                print(f'{1000 * (time.time() - start_time):.1f} ms')
    finally:
        image_client.close()
        cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-only', action='store_true')
    parser.add_argument('--port1', type=int, default=6000)
    parser.add_argument('--port2', type=int, default=6001)
    parser.add_argument('--scale-factor', type=float, default=0.35)
    parser.add_argument('--benchmark', action='store_true')
    main(parser.parse_args())
