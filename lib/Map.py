import cv2
import numpy as np
from lib.Car import Car

class Map:
    def __init__(self, color_path, map_path):
        self.color_image = cv2.imread(color_path, cv2.IMREAD_COLOR)
        self.map_image = cv2.imread(map_path, cv2.IMREAD_COLOR)

    def genarate_reset_vectors(self):

        vectors = []

        car = Car(self)
        car.max_steering_accel = np.inf

        spawns = self.get_spawns()

        pos = np.mean(spawns, axis=0)

        car.position = pos.astype(np.float32)
        car.angle = -np.pi / 2.0

        hit_finish = False

        angles = []
        count = 5
        for i in range(count):
            angles.append((np.pi/2) - i * np.pi/2/count)

        print(angles)

        while not hit_finish:
            hit_wall, hit_finish = car.update(5/1000)

            left_rays = [ car.cast_ray(-ang).distance for ang in angles ]
            right_rays = [ car.cast_ray(ang).distance for ang in angles ]

            centering = np.sum(left_rays) - np.sum(right_rays)

            if (car.forward_velocity < 2000):
                car.throttle = 1
            else:
                car.throttle = -.1

            car.steer = -centering * 0.01

            vectors.append([car.position[0], car.position[1], car.angle])
            cv2.circle(self.color_image, (int(car.position[0]), int(car.position[1])), 5, (0, 255, 0), -1)

        self.reset_vectors = np.array(vectors)

        return self.reset_vectors

    def reset_car(self, car):

        # Find closest reset vector
        distances = np.linalg.norm(self.reset_vectors[:, :2] - car.position, axis=1)

        closest = np.argmin(distances)

        car.position = np.array(self.reset_vectors[closest, :2])
        car.angle = np.array(self.reset_vectors[closest, 2])
        car.velocity = np.array([0., 0.])
        car.drifting = False
        car.disabled_for_ms = 2000




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

        