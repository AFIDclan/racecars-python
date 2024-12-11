import cv2
import numpy as np
from Map import Map
from Car1 import Car




track_map = cv2.imread("tracks/1_map.png", cv2.IMREAD_GRAYSCALE)
track_color = cv2.imread("tracks/1_color.png", cv2.IMREAD_COLOR)

map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

car = Car()

car.position = np.array(spawns[0]).astype(np.float32)
car.angle = -np.pi / 2.0


cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

step = 0
while True:

    disp_img = track_color.copy()   

    car.update(5/1000, disp_img)

    car.render(disp_img)

    step += 1
    
    car.throttle = 1

    if (step > 72 and step < 92):
        car.steer = -1
    else:
        car.steer = 0

    cv2.imshow("Racecar", disp_img)

    key = cv2.waitKey(5)

    # Exit on escape
    if key == 27:
        break