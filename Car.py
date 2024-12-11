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
        self.last_angle = 0

        self.vel_x = 0
        self.vel_y = 0

        self.accel_x = 0
        self.accel_y = 0

        self.max_velocity = 2200.0

        self.max_turn_rate = 1.0

        self.max_engine_accel = 1000.0
        self.max_lateral_accel = 5.0

        self.throttle = 0
        self.steer = 0


    def update(self, dt):

        throttle_chi = max(-1, min(1, self.throttle))

        throttle_accel_forward = throttle_chi * self.max_engine_accel

        steer_chi = max(-1, min(1, self.steer))
        self.angle += steer_chi * self.max_turn_rate * self.velocity * dt

        turn_delta = self.angle - self.last_angle

        turn_accel_lateral = np.sin(np.radians(turn_delta)) * self.velocity

        turn_accel_lateral = max(-self.max_lateral_accel, min(self.max_lateral_accel, turn_accel_lateral))

        accel_car_space = np.array([throttle_accel_forward, turn_accel_lateral])

        if (np.linalg.norm(accel_car_space) > self.max_engine_accel):
            accel_car_space = accel_car_space / np.linalg.norm(accel_car_space) * self.max_engine_accel

        rot_mat = np.array([
            [np.cos(np.radians(self.angle)), -np.sin(np.radians(self.angle))],
            [np.sin(np.radians(self.angle)), np.cos(np.radians(self.angle))]
        ])

        accel_world_space = accel_car_space @ rot_mat

        self.vel_x += accel_world_space[0]
        self.vel_y += accel_world_space[1]

        self.vel_x = max(-self.max_velocity, min(self.max_velocity, self.vel_x))
        self.vel_y = max(-self.max_velocity, min(self.max_velocity, self.vel_y))

        self.x += self.vel_x * dt
        self.y += self.vel_y * dt

        self.last_angle = self.angle
        


    @property
    def velocity(self):
        return np.sqrt(self.vel_x**2 + self.vel_y**2)

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
