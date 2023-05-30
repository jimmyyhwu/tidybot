# Author: Jimmy Wu
# Date: February 2023

import argparse
import time
from multiprocessing import Process
from multiprocessing.connection import Client
import cv2 as cv
from camera_server import CameraServer
from constants import CAMERA_SERIALS
from constants import CONN_AUTHKEY
from image_client import ImageClient

def main(args):
    # Start camera servers
    def start_camera_server(serial, port):
        CameraServer(serial, port=port).run()
    for serial, port in [(CAMERA_SERIALS[0], 6000), (CAMERA_SERIALS[1], 6001)]:
        Process(target=start_camera_server, args=(serial, port), daemon=True).start()

    # Wait for camera servers to be ready
    time.sleep(1.5)

    # Start image client
    image_client = ImageClient(port1=6000, port2=6001, scale_factor=0.35)

    conn = Client(('localhost', 6003), authkey=CONN_AUTHKEY)
    try:
        while True:
            if args.benchmark:
                start_time = time.time()

            image = image_client.get_image()
            #import random; import numpy as np
            #image_shape = (720, 1280, 3) if random.random() < 0.5 else (746, 640, 3)
            #image = np.random.randint(0, 256, image_shape, dtype=np.uint8)
            _, encoded_image = cv.imencode('.jpg', image)
            conn.send({'encoded_image': encoded_image, 'categories': ['object']})
            output = conn.recv()

            if args.benchmark:
                print(f'{1000 * (time.time() - start_time):.1f} ms')
            else:
                print(output)

    finally:
        image_client.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    main(parser.parse_args())
