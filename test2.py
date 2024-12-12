import cv2
import numpy as np
from Map import Map
from Car1 import Car




track_map = cv2.imread("tracks/1_map.png", cv2.IMREAD_GRAYSCALE)
track_color = cv2.imread("tracks/1_color.png", cv2.IMREAD_COLOR)

map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

car = Car( map )

car.position = np.array(spawns[0]).astype(np.float32)
car.angle = -np.pi / 2.0


cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

step = 0
while True:

    disp_img = track_color.copy()   

    car.update(5/1000, disp_img)

    car.render(disp_img, track_color)

    step += 1
    


    rays = [ car.cast_ray(ang) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]

    car.throttle = 1

    # if (car.forward_velocity < 1500):
    #     car.throttle = 1
    # else:
    #     car.throttle = -.1


    centering = rays[0].distance + rays[1].distance - rays[3].distance - rays[4].distance

    cv2.putText(disp_img, f"Centering: {centering}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(disp_img, f"Speed: {car.forward_velocity}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    car.steer = -centering * 0.05

    cv2.imshow("Racecar", disp_img)

    key = cv2.waitKey(5)

    # Exit on escape
    if key == 27:
        break