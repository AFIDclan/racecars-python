import cv2
import numpy as np
from Map import Map


track_map = cv2.imread("tracks/1_map.png", cv2.IMREAD_GRAYSCALE)
track_color = cv2.imread("tracks/1_color.png", cv2.IMREAD_COLOR)

map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

print(spawns)

cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("Racecar", track_color)

cv2.waitKey(0)