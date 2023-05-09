import socket
from multiprocessing.connection import Listener
from pathlib import Path
import cv2 as cv

def get_camera_params(serial):
    camera_params_file_path = Path('camera_params') / f'{serial}.yml'
    assert camera_params_file_path.exists()
    fs = cv.FileStorage(str(camera_params_file_path), cv.FILE_STORAGE_READ)
    image_width = int(fs.getNode('image_width').real())
    image_height = int(fs.getNode('image_height').real())
    camera_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('distortion_coefficients').mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs

def get_video_cap(serial, frame_width, frame_height):
    cap = cv.VideoCapture(f'/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0')
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Gives much better latency

    # Disable all auto
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv.CAP_PROP_AUTO_WB, 0)  # White balance
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = off, 3 = on

    # Read several frames to let settings (especially gain/exposure) stabilize
    for _ in range(30):
        cap.read()
        cap.set(cv.CAP_PROP_FOCUS, 0)  # Fixed focus
        cap.set(cv.CAP_PROP_TEMPERATURE, 3900)  # Fixed white balance
        cap.set(cv.CAP_PROP_GAIN, 10)  # Fixed gain
        cap.set(cv.CAP_PROP_EXPOSURE, 156)  # Fixed exposure

    # Check all settings match expected
    assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == frame_width
    assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == frame_height
    assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
    assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
    assert cap.get(cv.CAP_PROP_AUTO_WB) == 0
    assert cap.get(cv.CAP_PROP_AUTO_EXPOSURE) == 1
    assert cap.get(cv.CAP_PROP_FOCUS) == 0
    assert cap.get(cv.CAP_PROP_TEMPERATURE) == 3900
    assert cap.get(cv.CAP_PROP_GAIN) == 10
    assert cap.get(cv.CAP_PROP_EXPOSURE) == 156

    return cap

class Camera:
    def __init__(self, robot_idx):
        serial = {0: '634093BE', 1: '44251E9E', 2:'7E841E9E'}[robot_idx]
        image_width, image_height, camera_matrix, dist_coeffs = get_camera_params(serial)
        self.cap = get_video_cap(serial, image_width, image_height)
        self.map_x, self.map_y = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, camera_matrix, (image_width, image_height), cv.CV_32FC1)
        self.image_size = (image_width // 2, image_height // 2)

    def get_encoded_image(self):
        assert self.cap.get(cv.CAP_PROP_EXPOSURE) == 156
        assert self.cap.get(cv.CAP_PROP_GAIN) == 10
        assert self.cap.get(cv.CAP_PROP_TEMPERATURE) == 3900
        assert self.cap.get(cv.CAP_PROP_FOCUS) == 0

        # Read new frame
        image = None
        while image is None:
            _, image = self.cap.read()  # Flush buffered frame
            _, image = self.cap.read()

        # Undistort image
        image = cv.remap(image, self.map_x, self.map_y, cv.INTER_LINEAR)

        # Resize image
        image = cv.resize(image, self.image_size, cv.INTER_LINEAR)  # Resize to 50%

        # Encode as JPEG
        _, image = cv.imencode('.jpg', image)

        return image

    def disconnect(self):
        self.cap.release()

class CameraServer:
    def __init__(self, camera, hostname='0.0.0.0', port=6010):
        self.camera = camera
        self.listener = Listener((hostname, port), authkey=b'secret password')

    def run(self):
        address, port = self.listener.address
        while True:
            print(f'Waiting for connection ({address}:{port})')
            conn = self.listener.accept()
            print(f'Connected! ({address}:{port})')
            while True:
                try:
                    conn.recv()
                    conn.send(self.camera.get_encoded_image())
                except (ConnectionResetError, EOFError, BrokenPipeError):
                    break

def main():
    hostname = socket.gethostname()
    assert hostname.startswith('iprl-bot')
    robot_num = int(hostname[-1])
    robot_idx = robot_num - 1
    port = 6010 + robot_idx
    camera = Camera(robot_idx)
    try:
        CameraServer(camera, port=port).run()
    finally:
        camera.disconnect()

if __name__ == '__main__':
    main()
