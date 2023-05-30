import cv2 as cv

ROBOT_HOSTNAME_PREFIX = 'iprl-bot'
CONN_AUTHKEY = b'secret password'

################################################################################
# Board and markers

CHARUCO_BOARD_PARAMS = {
    'squares_x': 10,
    'squares_y': 7,
    'square_length': 0.024,  # 24 mm
    'marker_length': 0.018,  # 18 mm
}
MARKER_PARAMS = {
    'marker_length': 0.09,  # 90 mm
    'sticker_length': 0.12,  # 120 mm
}
MARKER_DICT_ID = cv.aruco.DICT_4X4_50
MARKER_IDS = [10, 13, 17, 21, 23, 30, 37, 41, 42, 45, 48, 49]

################################################################################
# Camera

CAMERA_SERIALS = [
    'E4298F4E',  # Top camera
    '099A11EE',  # Bottom camera
]
CAMERA_FOCUS = 0
CAMERA_TEMPERATURE = 3900
CAMERA_EXPOSURE = 77  # 77 is best, 156 is slightly worse, 312 results in motion blur
CAMERA_GAIN = 50  # Increments of 10
CAMERA_HEIGHT = 2.71  # 271 cm
PIXELS_PER_M = 500

################################################################################
# Floor

NUM_FLOOR_TILES_X = 6
NUM_FLOOR_TILES_Y = 7
FLOOR_TILE_SIZE = 24 * 0.0254  # 2 ft
FLOOR_LENGTH = NUM_FLOOR_TILES_Y * FLOOR_TILE_SIZE
FLOOR_WIDTH = NUM_FLOOR_TILES_X * FLOOR_TILE_SIZE

################################################################################
# Robot

NUM_ROBOTS = 3
ROBOT_WIDTH = 0.465  # 46.5 cm
ROBOT_HEIGHT = 0.325  # 32.5 cm
ROBOT_DIAG = 0.55  # 55 cm
