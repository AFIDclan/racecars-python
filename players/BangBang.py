from lib.Player import Player
import numpy as np

class BangBang(Player):

    def cast_rays(self):
        return [ self.car.cast_ray(ang, 5, 400) for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ] ]
    

    def control(self, rays):
        self.car.throttle = 1

        if (self.car.forward_velocity < 165 or rays[2].distance > 250):
            self.car.throttle = 1
        else:
            self.car.throttle = -1

        centering = rays[0].distance + rays[1].distance - rays[3].distance - rays[4].distance

        self.car.steer = -centering * 0.02