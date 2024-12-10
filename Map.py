import cv2
import numpy as np

class Map:
    def __init__(self, color_path, map_path):
        self.color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        self.map_image = cv2.imread(map_path, cv2.IMREAD_COLOR)

    def get_spawns(self):
        
        ## Spawn points are blue
        blue = np.array([255, 0, 0])
        mask = np.all(self.map_image == blue, axis=2)

        ## Find the coordinates of the blue pixels
        y, x = np.where(mask)

        spawns = []

        for i in range(len(x)):
            spawns.append((x[i], y[i]))

        return spawns

        