import time
from multiprocessing import Process
from camera_server import CameraServer
from constants import CAMERA_SERIALS

def start_camera_server(serial, port):
    CameraServer(serial, port=port).run()

def main():
    for serial, port in [(CAMERA_SERIALS[0], 6000), (CAMERA_SERIALS[1], 6001)]:
        Process(target=start_camera_server, args=(serial, port), daemon=True).start()
    while True:
        time.sleep(1)

main()
