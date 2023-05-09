import argparse
import time
from multiprocessing.connection import Client
from pathlib import Path
from queue import Queue
from threading import Thread
import cv2 as cv

def handle_object_detection(queue):
    try:
        object_detector_conn = Client(('localhost', 6003), authkey=b'secret password')
        categories = ['clothing', 'toy']
    except ConnectionRefusedError:
        object_detector_conn = None

    while True:
        if not queue.empty():
            encoded_image = queue.get()
            if object_detector_conn is not None:
                object_detector_conn.send({'encoded_image': encoded_image, 'categories': categories, 'use_clip': True})
                print(object_detector_conn.recv())
        time.sleep(0.001)

def main(args):
    hostname = f'iprl-bot{args.robot_num}'
    robot_idx = args.robot_num - 1
    port = 6010 + robot_idx
    window_name = 'out'
    conn = Client((hostname, port), authkey=b'secret password')

    queue = Queue()
    Thread(target=handle_object_detection, args=(queue,), daemon=True).start()

    try:
        last_read_time = 0
        while True:
            # Read new image
            if time.time() - last_read_time > 1:
                conn.send(None)
                encoded_image = conn.recv()
                last_read_time = time.time()
                image = cv.imdecode(encoded_image, cv.IMREAD_COLOR)
                queue.put(encoded_image)

            # Show current image
            cv.imshow(window_name, image)
            key = cv.waitKey(1)
            if key == 27 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 0.5:
                break
            if key == ord('s'):  # Save image
                image_dir = Path('images')
                if not image_dir.exists():
                    image_dir.mkdir()
                image_path = image_dir / f'image-{int(10 * time.time()) % 100000000}.jpg'
                with open(image_path, 'wb') as f:
                    f.write(encoded_image)
                print(f'Saved image to {image_path}')
    finally:
        cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot-num', type=int, default=1)
    main(parser.parse_args())
