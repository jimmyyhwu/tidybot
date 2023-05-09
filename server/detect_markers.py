# Adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/detect_markers.cpp
import argparse
import cv2 as cv
import utils
from constants import MARKER_DICT_ID

def main(serial):
    # Read camera parameters
    image_width, image_height, camera_matrix, dist_coeffs = utils.get_camera_params(serial)

    # Set up webcam
    cap = utils.get_video_cap(serial, image_width, image_height)

    # Set up aruco dict
    aruco_dict = cv.aruco.Dictionary_get(MARKER_DICT_ID)

    window_name = 'out'
    cv.namedWindow(window_name)
    while True:
        # Fixed gain/exposure can be unreliable
        assert cap.get(cv.CAP_PROP_GAIN) == 50
        assert cap.get(cv.CAP_PROP_EXPOSURE) == 77

        if cv.waitKey(1) == 27 or cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 0.5:
            break

        image = None
        while image is None:
            _, image = cap.read()

        # Undistort image and detect markers
        image = cv.undistort(image, camera_matrix, dist_coeffs)
        corners, ids, _ = cv.aruco.detectMarkers(image, aruco_dict)

        # Show detections
        image_copy = image.copy()
        if ids is not None:
            cv.aruco.drawDetectedMarkers(image_copy, corners, ids)
        cv.imshow(window_name, image_copy)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default=None)
    parser.add_argument('--camera2', action='store_true')  # Use bottom camera
    args = parser.parse_args()
    if args.serial is None:
        if args.camera2:
            args.serial = '099A11EE'
        else:
            args.serial = 'E4298F4E'
    main(args.serial)
