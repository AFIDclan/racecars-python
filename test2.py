import cv2
import numpy as np
from Map import Map
from Car import Car




track_map = cv2.imread("tracks/1_map.png", cv2.IMREAD_GRAYSCALE)
track_color = cv2.imread("tracks/1_color.png", cv2.IMREAD_COLOR)

map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

car = Car(map, None)

car.x = spawns[0][0]
car.y = spawns[0][1]

car.angle = 25

car2 = Car(map, None)

car2.x = spawns[1][0]
car2.y = spawns[1][1]

car2.angle = 0


print(spawns)

cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    car.update()
    car2.update()

    disp_img = track_color.copy()   

    car.render(disp_img)
    car2.render(disp_img)

    car.throttle = 1
    car.angle -= 1

    cv2.imshow("Racecar", disp_img)

    key = cv2.waitKey(5)

    # Exit on escape
    if key == 27:
        break