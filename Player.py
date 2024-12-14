from Car import Car
import numpy as np
import time

class Player:
    def __init__(self, map, color):
        self.car = Car(map, color)
        self.lap_times = []
        self.current_lap_start = time.time()
        self.cleared_finish = False

    def cast_rays(self):
        return [ self.car.cast_ray(ang, 2, 400) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]
    

    def control(self, rays):

        self.car.throttle = 1

        if (self.car.forward_velocity < 150 or rays[2].distance > 350):
            self.car.throttle = 1
        else:
            self.car.throttle = -.1

        centering = rays[0].distance + rays[1].distance - rays[3].distance - rays[4].distance

        self.car.steer = -centering * 0.05