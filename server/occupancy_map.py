# Author: Jimmy Wu
# Date: February 2023

import math
import cv2 as cv
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.draw import line
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import flood_fill
import utils
from constants import FLOOR_LENGTH, FLOOR_WIDTH, PIXELS_PER_M, ROBOT_DIAG
from shortest_paths.shortest_paths import GridGraph

class OccupancyMap:
    def __init__(self, obstacles=None, scale_factor=0.1, debug=False):
        self.scale_factor = scale_factor

        # Binary occupancy map showing where obstacles are
        image_width = round(scale_factor * PIXELS_PER_M * FLOOR_WIDTH)
        image_height = round(scale_factor * PIXELS_PER_M * FLOOR_LENGTH)
        self.occupancy_map = np.zeros((image_height, image_width), dtype=np.uint8)

        # Add obstacles
        if obstacles is not None:
            for obstacle in obstacles:
                pos_x, pos_y = obstacle['position']
                dim_x, dim_y = obstacle['dimensions']
                min_xy = self.position_to_pixel_xy((pos_x - dim_x / 2, pos_y + dim_y / 2))
                max_xy = self.position_to_pixel_xy((pos_x + dim_x / 2, pos_y - dim_y / 2))
                self.occupancy_map[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]] = 1

        # Configuration space
        pixels_per_m = self.scale_factor * PIXELS_PER_M
        self.robot_footprint = disk(math.ceil(pixels_per_m * (ROBOT_DIAG + 0.20) / 2))  # 20 cm extra buffer around robot

        # Visualization
        self.debug = debug
        if self.debug:
            self.window_name_occupancy_map = 'Occupancy Map'
            self.window_name_cspace = 'Configuration Space'
            cv.namedWindow(self.window_name_occupancy_map)
            cv.namedWindow(self.window_name_cspace)
            cv.moveWindow(self.window_name_occupancy_map, 800, 100)
            cv.moveWindow(self.window_name_cspace, 800, 500)

    def position_to_pixel_xy(self, position):
        return utils.position_to_pixel_xy(position, self.occupancy_map.shape, self.scale_factor)

    def pixel_xy_to_position(self, pixel_xy):
        return utils.pixel_xy_to_position(pixel_xy, self.occupancy_map.shape, self.scale_factor)

    def shortest_path(self, source_position, target_position):
        # Convert positions to pixel indices
        source_j, source_i = self.position_to_pixel_xy(source_position)
        target_j, target_i = self.position_to_pixel_xy(target_position)

        # Flood fill target to handle case where target position is inside obstacle
        occupancy_map = flood_fill(self.occupancy_map, (target_i, target_j), 0)
        assert occupancy_map.dtype == np.uint8

        # Add boundaries of floor space
        occupancy_map[:, :1] = 1
        occupancy_map[:, -1:] = 1
        occupancy_map[:1, :] = 1
        occupancy_map[-1:, :] = 1

        if self.debug:
            cv.imshow(self.window_name_occupancy_map, 255 * occupancy_map)
            #cv.imwrite('occupancy-map.png', 255 * occupancy_map)

        # Generate configuration space by dilating the occupancy map
        configuration_space = 1 - binary_dilation(occupancy_map, self.robot_footprint).astype(np.uint8)
        assert configuration_space.dtype == np.uint8

        # Fill in regions that cannot be reached
        contours, _ = cv.findContours(configuration_space, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Find all connected components
        if len(contours) > 1:
            source_idx = None
            for idx, contour in enumerate(contours):
                if cv.pointPolygonTest(contour, (source_j, source_i), False) >= 0:  # Find component containing source position
                    source_idx = idx
                    break
            if source_idx is not None:  # If source_idx is None then robot might be inside an obstacle
                for idx, contour in enumerate(contours):
                    if idx != source_idx:
                        cv.drawContours(configuration_space, contours, idx, 0, -1)  # Fill in component

        if self.debug:
            cv.imshow(self.window_name_cspace, 255 * configuration_space)
            #cv.imwrite('configuration-space.png', 255 * configuration_space)

        # First check if there is a straight line path
        rr, cc = line(source_i, source_j, target_i, target_j)
        if (1 - configuration_space[rr, cc]).sum() == 0:
            return [source_position, target_position]

        # Run SPFA to find shortest path
        grid_graph = GridGraph(configuration_space)
        closest_cspace_indices = distance_transform_edt(1 - configuration_space, return_distances=False, return_indices=True)
        source_i, source_j = closest_cspace_indices[:, source_i, source_j]
        target_i, target_j = closest_cspace_indices[:, target_i, target_j]
        path_pixel_indices = grid_graph.shortest_path((source_i, source_j), (target_i, target_j))

        # Convert pixel indices back to positions
        path = [self.pixel_xy_to_position((j, i)) for i, j in path_pixel_indices]
        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position

        return path

def main():
    scenario = utils.load_yaml('scenarios/test.yml')
    occupancy_map = OccupancyMap(obstacles=scenario['receptacles'].values())
    print(occupancy_map.shortest_path((-1, 1), (1, 1)))
    print(occupancy_map.shortest_path((-1, 1), (1, -1)))

if __name__ == '__main__':
    main()
