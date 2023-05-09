# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from multiprocessing import shared_memory
from queue import Queue
from threading import Thread
import cv2 as cv
import numpy as np
import utils
from publisher import Publisher

class CameraServer(Publisher):
    def __init__(self, serial, **kwargs):
        super().__init__(**kwargs)

        # Set up video cap
        image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(serial)
        self.cap = utils.get_video_cap(serial, image_width, image_height)  # 1540 ms
        self.last_read_time = time.time()
        self.queue = Queue(maxsize=1)
        Thread(target=self.camera_worker, daemon=True).start()

        # Compute maps for undistort
        self.map_x, self.map_y = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (image_width, image_height), cv.CV_32FC1)

        # Set up shared memory array
        image = self.get_image()
        self.shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
        self.image_shm = np.ndarray(image.shape, dtype=image.dtype, buffer=self.shm.buf)

    def get_image(self):
        # Read new frame
        image = None
        while image is None:
            _, image = self.cap.read()  # 5 to 17 ms

        # Undistort image
        image = cv.remap(image, self.map_x, self.map_y, cv.INTER_LINEAR)  # 2 to 5 ms

        return image

    def camera_worker(self):
        while True:
            if self.queue.empty():
                image = self.get_image()
                self.queue.put((time.time(), image))
                assert self.cap.get(cv.CAP_PROP_EXPOSURE) == 77
                assert self.cap.get(cv.CAP_PROP_GAIN) == 50
                assert self.cap.get(cv.CAP_PROP_TEMPERATURE) == 3900
                assert self.cap.get(cv.CAP_PROP_FOCUS) == 0
            time.sleep(0.0001)

    def get_data(self):
        # Reading new frames too quickly causes latency spikes
        while time.time() - self.last_read_time < 0.0333:  # 30 fps
            time.sleep(0.0001)

        capture_time, image = self.queue.get()
        if time.time() - capture_time > 0.1:  # 100 ms
            self.queue.get()  # Flush camera buffer
            _, image = self.queue.get()
        self.last_read_time = time.time()
        np.copyto(self.image_shm, image)  # 0.2 ms
        return {'name': self.shm.name, 'shape': image.shape, 'dtype': image.dtype}

    def clean_up(self):
        self.cap.release()
        self.shm.close()
        self.shm.unlink()

def main(args):
    if args.serial is None:
        if args.camera2:
            args.serial = '099A11EE'
        else:
            args.serial = 'E4298F4E'
    CameraServer(args.serial, port=args.port).run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=None)
    parser.add_argument('--camera2', action='store_true')  # Use bottom camera
    parser.add_argument('--port', type=int, default=6000)
    main(parser.parse_args())
