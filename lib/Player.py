from lib.Car import Car
import numpy as np
import time

class Player:
    def __init__(self, map, color):
        self.car = Car(map, color)
        self.lap_times = []
        self.current_lap_start = 0
        self.cleared_finish = False

    def cast_rays(self):
        return [ self.car.cast_ray(ang, 5, 400) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]
    

    def control(self, rays):
        pass