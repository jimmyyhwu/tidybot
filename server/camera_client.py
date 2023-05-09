# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import Client
import cv2 as cv
import numpy as np

# See https://bugs.python.org/issue38119
def remove_shm_from_resource_tracker():
    # pylint: disable=all
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(self, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(self, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]

class CameraClient:
    def __init__(self, port):
        # Connect to camera server
        self.conn = Client(('localhost', port), authkey=b'secret password')

        # Set up shared memory block for reading camera images
        self.conn.send(None)
        data = self.conn.recv()
        remove_shm_from_resource_tracker()
        self.shm = shared_memory.SharedMemory(name=data['name'])
        self.image = np.ndarray(data['shape'], dtype=data['dtype'], buffer=self.shm.buf)
        self.conn.send(None)

    def get_image(self):
        self.conn.recv()
        self.conn.send(None)
        return self.image

    def close(self):
        self.shm.close()

def main(args):
    camera_client = CameraClient(args.port)
    window_name = 'out'

    try:
        while True:
            if args.benchmark:
                start_time = time.time()

            image = camera_client.get_image()

            if not args.benchmark:
                cv.imshow(window_name, image)
                if cv.waitKey(1) == 27 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 0.5:
                    break

            if args.benchmark:
                print(f'{1000 * (time.time() - start_time):.1f} ms')
    finally:
        camera_client.close()
        cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--benchmark', action='store_true')
    main(parser.parse_args())
