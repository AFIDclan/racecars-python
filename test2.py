import cv2
import numpy as np
from Map import Map
from Car import Car
import time

def homo_apply(vec, H):
    vec = np.array([vec[0], vec[1], 1])

    out = H @ vec

    out[0] /= out[2]
    out[1] /= out[2]

    return out[:2]



map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

try:
    vectors = np.load('tracks/1_reset_vectors.npy')
    map.reset_vectors = vectors
except Exception as e:
    print("No reset vectors found for map. Genarating them.")
    vectors = map.genarate_reset_vectors()
    np.save("tracks/1_reset_vectors.npy", vectors)

car = Car( map )

car.position = np.array(spawns[0]).astype(np.float32)
car.angle = -np.pi / 2.0


cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

step = 0

last = time.time()

while True:

    disp_img = map.color_image.copy()   

    delta = (time.time() - last)
    last = time.time()
    print(delta)
    hit_wall, hit_finish = car.update(delta)

    if (hit_wall):
        map.reset_car(car)
        

    car.render(disp_img, map.color_image)

    step += 1
    


    rays = [ car.cast_ray(ang) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]

    car.throttle = 1

    if (car.forward_velocity < 300):
        car.throttle = 1
    else:
        car.throttle = -.1


    centering = rays[0].distance + rays[1].distance - rays[3].distance - rays[4].distance

    cv2.putText(disp_img, f"Centering: {centering}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(disp_img, f"Speed: {car.forward_velocity}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    car.steer = -centering * 0.05

    cv2.imshow("Racecar", disp_img)

    key = cv2.waitKey(5)

    # Exit on escape
    if key == 27:
        break