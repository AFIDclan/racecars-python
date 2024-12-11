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
        angle_delta = self.angle - self.last_angle 

        # Calculate turning radius
        angle_circum = abs(((np.pi*2) / angle_delta) * local_velocity[1] * dt)
        angle_radius = angle_circum / np.pi / 2.0
        angle_radius_inv = 0. if abs(angle_delta) < 1e-6 or angle_radius < 1e-6 else 1 / angle_radius

        acceleration = np.array([
            local_velocity[1]**2 * angle_radius_inv * -np.sign(steer),    # Centripetal acceleration
            throttle * self.max_throttle_accel          # Throttle acceleration
        ])
        

        circle_pos = homo_apply([ angle_radius* -np.sign(steer), 0 ], self.H_C2G).astype(np.int32)
        
        if not np.any(np.isnan(circle_pos)) and math.isfinite(angle_radius):
            print(acceleration)
            cv2.circle(debug_image, circle_pos, int(angle_radius), (0,255,255), 2) 


        acceleration[0] = max(-self.max_steering_accel, min(self.max_steering_accel, acceleration[0]))

        # Convert acceleration to global
        global_acceleration = homo_rotate(acceleration, self.H_C2G)

        self.velocity += global_acceleration * dt

        self.position += self.velocity * dt

        self.last_angle = self.angle


    def render(self, image):
        # TODO: Make this draw an image instead

        rect = [
            [-self.width / 2, -self.height / 2],
            [self.width / 2, -self.height / 2],
            [self.width / 2, self.height / 2],
            [-self.width / 2, self.height / 2]
        ]
        
        H = self.H_C2G

        world_rect = np.array([homo_apply(v, H) for v in rect]).astype(np.int32)
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
