import cv2
import numpy as np


class Car:
    def __init__(self, map, player):
        self.map = map
        self.player = player

        self.width = 30
        self.height = 20
        
        self.x = None
        self.y = None
        self.angle = None
        self.velocity = 0
        self.max_velocity = 5

        self.max_pos_accel = 0.1
        self.max_neg_accel = 0.1

        self.throttle = 0


    def update(self):
        throttle_chi = max(-1, min(1, self.throttle))

        if throttle_chi > 0:
            self.velocity += self.max_pos_accel * throttle_chi
        else:
            self.velocity += self.max_neg_accel * throttle_chi

        self.velocity = max(0, min(self.max_velocity, self.velocity))

        self.x += self.velocity * np.cos(np.radians(self.angle))
        self.y += self.velocity * np.sin(np.radians(self.angle))

        


    def render(self, image):
        # TODO: Make this draw an image instead

        rect = np.array([
            [-self.width / 2, -self.height / 2],
            [self.width / 2, -self.height / 2],
            [self.width / 2, self.height / 2],
            [-self.width / 2, self.height / 2]
        ])

        rot_mat = np.array([
            [np.cos(np.radians(self.angle)), -np.sin(np.radians(self.angle))],
            [np.sin(np.radians(self.angle)), np.cos(np.radians(self.angle))]
        ])

        rot_rect = rect @ rot_mat.T

        final_rect = rot_rect + np.array([self.x, self.y])
        final_rect = final_rect.astype(np.int32)

        cv2.fillConvexPoly(image, final_rect, (0, 255, 0))
