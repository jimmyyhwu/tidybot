# Adapted from https://github.com/opencv/opencv_contrib/blob/master/modules/aruco/samples/calibrate_camera_charuco.cpp
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import utils
from constants import CHARUCO_BOARD_PARAMS, MARKER_DICT_ID

def main(args):
    # Set up webcam
    image_width = args.image_width
    image_height = args.image_height
    cap = utils.get_video_cap(args.serial, image_width, image_height)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    assert cap.get(cv2.CAP_PROP_FOCUS) == 0

    # Set up aruco dict and board
    aruco_dict = cv2.aruco.Dictionary_get(MARKER_DICT_ID)
    board = cv2.aruco.CharucoBoard_create(
        CHARUCO_BOARD_PARAMS['squares_x'], CHARUCO_BOARD_PARAMS['squares_y'], CHARUCO_BOARD_PARAMS['square_length'], CHARUCO_BOARD_PARAMS['marker_length'], aruco_dict)

    # Enable corner refinement
    detector_params = cv2.aruco.DetectorParameters_create()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Capture images
    all_corners = []
    all_ids = []
    all_imgs = []
    while True:
        _, image = cap.read()
        if image is None:
            continue

        # Detect markers
        corners, ids, rejected = cv2.aruco.detectMarkers(image, aruco_dict, parameters=detector_params)

        # Refine markers
        corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(image, board, corners, ids, rejected)

        # Interpolate corners
        if ids is not None:
            _, curr_charuco_corners, curr_charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board)

        # Draw results
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(image_copy, corners)

            if curr_charuco_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(image_copy, curr_charuco_corners, curr_charuco_ids)

        # Display and wait for keyboard input
        cv2.putText(image_copy, "Press 'c' to add current frame. 'ESC' to finish and calibrate", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('out', image_copy)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        if key == ord('c') and ids is not None and len(ids) > 4:
            print('Frame captured')
            all_corners.append(corners)
            all_ids.append(ids)
            all_imgs.append(image)

    cap.release()
    cv2.destroyAllWindows()

    if len(all_imgs) < 1:
        print('Not enough captures for calibration')
        sys.exit()

    # Aruco calibration
    all_corners_concatenated = []
    all_ids_concatenated = []
    marker_counter_per_frame = []
    for corners, ids in zip(all_corners, all_ids):
        marker_counter_per_frame.append(len(corners))
        all_corners_concatenated.extend(corners)
        all_ids_concatenated.extend(ids)
    all_corners_concatenated = np.asarray(all_corners_concatenated)
    all_ids_concatenated = np.asarray(all_ids_concatenated)
    marker_counter_per_frame = np.asarray(marker_counter_per_frame)
    rep_error_aruco, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraAruco(all_corners_concatenated, all_ids_concatenated, marker_counter_per_frame, board, (image_width, image_height), None, None)

    # Charuco calibration using previous camera params
    all_charuco_corners = []
    all_charuco_ids = []
    for corners, ids, image in zip(all_corners, all_ids, all_imgs):
        _, curr_charuco_corners, curr_charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, board, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)
        all_charuco_corners.append(curr_charuco_corners)
        all_charuco_ids.append(curr_charuco_ids)
    if len(all_charuco_corners) < 4:
        print('Not enough corners for calibration')
        sys.exit()
    rep_error, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, (image_width, image_height), camera_matrix, dist_coeffs)

    print('Rep Error:', rep_error)
    print('Rep Error Aruco:', rep_error_aruco)

    # Save camera params
    camera_params_file_path = Path('camera_params') / f'{args.serial}.yml'
    camera_params_dir = camera_params_file_path.parent
    if not camera_params_dir.exists():
        camera_params_dir.mkdir()
    fs = cv2.FileStorage(str(camera_params_file_path), cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', image_width)
    fs.write('image_height', image_height)
    fs.write('camera_matrix', camera_matrix)
    fs.write('distortion_coefficients', dist_coeffs)
    fs.write('avg_reprojection_error', rep_error)
    fs.release()
    print('Calibration saved to ', camera_params_file_path)

    # Show interpolated charuco corners
    for image, ids, charuco_corners, charuco_ids in zip(all_imgs, all_ids, all_charuco_corners, all_charuco_ids):
        image_copy = image.copy()
        if ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)
        cv2.imshow('out', image_copy)
        key = cv2.waitKey(0)
        if key == 27:  # Esc key
            break

parser = argparse.ArgumentParser()
parser.add_argument('--serial', default='E4298F4E')
parser.add_argument('--image-width', type=int, default=1280)
parser.add_argument('--image-height', type=int, default=720)
main(parser.parse_args())
