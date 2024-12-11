import numpy as np
import cv2
import math

def homo_apply(vec, H):
    vec = np.array([vec[0], vec[1], 1])

    out = H @ vec

    out[0] /= out[2]
    out[1] /= out[2]

    return out[:2]

def homo_rotate(vec, H):
    vec = np.array([vec[0], vec[1]])
    return H[:2, :2] @ vec


class Car:
    def __init__(self):

        self.position = [ 0, 0 ]
        self.velocity = [ 0, 0 ]

        self.max_forward_velocity = 2000
        self.max_rad_per_vel = 0.00008

        self.max_throttle_accel = 3500
        self.max_steering_accel = 10000


        self.angle = 0
        self.last_angle = 0

        self.width = 20
        self.height = 30

        self.throttle = 0
        self.steer = 0


    def update(self, dt, debug_image):

        # Sainitize
        steer = max(-1, min(1, self.steer))
        throttle = max(-1, min(1, self.throttle))
        
        # Convert velocity to local
        local_velocity = homo_rotate(self.velocity, self.H_G2C)

        # Stop accelerating if we hit max speed
        if (throttle > 0 and local_velocity[1] > self.max_forward_velocity):
            throttle = 0

        if (throttle < 0 and local_velocity[1] < -self.max_forward_velocity):
            throttle = 0


        # Turn car at a ratio of the velocity forward
        self.angle += local_velocity[1] * steer * self.max_rad_per_vel

        acceleration = np.array([
            -local_velocity[0] / dt,
            throttle * self.max_throttle_accel          # Throttle acceleration
        ])
        
        clipped_lateral_accel = max(-self.max_steering_accel, min(self.max_steering_accel, acceleration[0]))

        self.drifting = clipped_lateral_accel != acceleration[0]

        acceleration[0] = clipped_lateral_accel

        # Convert acceleration to global
        global_acceleration = homo_rotate(acceleration, self.H_C2G)

        self.velocity += global_acceleration * dt

        self.position += self.velocity * dt



    def render(self, image, static_image):
        # TODO: Make this draw an image instead

        rect = [
            [-self.width / 2, -self.height / 2],
            [self.width / 2, -self.height / 2],
            [self.width / 2, self.height / 2],
            [-self.width / 2, self.height / 2]
        ]
        
        H = self.H_C2G

        world_rect = np.array([homo_apply(v, H) for v in rect]).astype(np.int32)

        if (self.drifting):
            for corner in world_rect:
                cv2.circle(static_image, corner, 4, (0, 0, 0), -1)

        cv2.fillConvexPoly(image, world_rect, (0, 255, 0))



    @property
    def H_C2G(self):
        return np.array([
            [np.cos(self.angle), -np.sin(self.angle), self.position[0]],
            [np.sin(self.angle), np.cos(self.angle), self.position[1]],
            [0., 0., 1.]
        ])
    
    @property
    def H_G2C(self):
        return np.linalg.inv(self.H_C2G)
