import time
from multiprocessing import Process
from camera_server import CameraServer

def start_camera_server(serial, port):
    CameraServer(serial, port=port).run()
for serial, port in [('E4298F4E', 6000), ('099A11EE', 6001)]:
    Process(target=start_camera_server, args=(serial, port), daemon=True).start()

while True:
    time.sleep(1)
