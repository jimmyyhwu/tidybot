import json
import math
import sys
from pathlib import Path
import cv2 as cv
import yaml
from constants import PIXELS_PER_M

################################################################################
# Camera

def get_video_cap(serial, frame_width, frame_height):
    if sys.platform == 'darwin':
        return cv.VideoCapture(0)
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
        cap.set(cv.CAP_PROP_GAIN, 50)  # Fixed gain
        cap.set(cv.CAP_PROP_EXPOSURE, 77)  # Fixed exposure

    # Check all settings match expected
    assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == frame_width
    assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == frame_height
    assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
    assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
    assert cap.get(cv.CAP_PROP_AUTO_WB) == 0
    assert cap.get(cv.CAP_PROP_AUTO_EXPOSURE) == 1
    assert cap.get(cv.CAP_PROP_FOCUS) == 0
    assert cap.get(cv.CAP_PROP_TEMPERATURE) == 3900
    assert cap.get(cv.CAP_PROP_GAIN) == 50
    assert cap.get(cv.CAP_PROP_EXPOSURE) == 77

    return cap

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

def get_camera_alignment_params(serial):
    params_path = Path('camera_params') / f'{serial}.json'
    assert params_path.exists()
    with open(params_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    return labels['camera_center'], labels['camera_corners']

################################################################################
# YAML

def load_yaml(path):
    with open(path, 'r', encoding='utf8') as f:
        return yaml.safe_load(f)

################################################################################
# Printouts

def get_paper_params(orientation='P'):
    width, height, margin = 8.5, 11, 0.5
    if orientation == 'L':
        width, height = height, width
    ppi = 600
    mm_per_in = 25.4
    params = {}
    params['width_mm'] = mm_per_in * width
    params['height_mm'] = mm_per_in * height
    params['margin_mm'] = mm_per_in * margin
    params['mm_per_printed_pixel'] = mm_per_in / ppi
    return params

################################################################################
# Math

def distance(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)

def restrict_heading_range(h):
    return (h + math.pi) % (2 * math.pi) - math.pi

def clip(a, a_min, a_max):
    return min(a_max, max(a, a_min))

def position_to_pixel_xy(position, image_shape, scale_factor):
    pixels_per_m = scale_factor * PIXELS_PER_M
    pixel_x = math.floor(image_shape[1] / 2 + position[0] * pixels_per_m)
    pixel_y = math.floor(image_shape[0] / 2 - position[1] * pixels_per_m)
    pixel_x = clip(pixel_x, 0, image_shape[1] - 1)
    pixel_y = clip(pixel_y, 0, image_shape[0] - 1)
    return pixel_x, pixel_y

def pixel_xy_to_position(pixel_xy, image_shape, scale_factor):
    pixels_per_m = scale_factor * PIXELS_PER_M
    position_x = ((pixel_xy[0] + 0.5) - image_shape[1] / 2) / pixels_per_m
    position_y = (image_shape[0] / 2 - (pixel_xy[1] + 0.5)) / pixels_per_m
    return position_x, position_y
