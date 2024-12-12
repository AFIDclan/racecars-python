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

class Ray:
    def __init__(self, angle, did_hit, distance, hit_point):

        self.angle = angle
        self.did_hit = did_hit
        self.distance = distance
        self.hit_point = hit_point


class Car:
    def __init__(self, map):

        self.position = [ 0, 0 ]
        self.velocity = [ 0, 0 ]

        self.map = map

        self.max_forward_velocity = 2500
        self.max_rad_per_vel = 0.00008

        self.max_throttle_accel = 5500
        self.max_brake_accel = 10000
        self.max_steering_accel = 15000


        self.angle = 0
        self.last_angle = 0

        self.width = 20
        self.height = 30

        self.throttle = 0
        self.steer = 0
        self.rays = []


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

        acceleration = None
        
        if throttle < 0:

            # Braking
            acceleration = np.array([
                -local_velocity[0] / dt,
                max(-local_velocity[1] / dt, -self.max_brake_accel)
            ])
        else:

            # Throttling
            acceleration = np.array([
                -local_velocity[0] / dt,
                throttle * self.max_throttle_accel
            ])
        
        clipped_lateral_accel = max(-self.max_steering_accel, min(self.max_steering_accel, acceleration[0]))

        self.drifting = clipped_lateral_accel != acceleration[0]

        acceleration[0] = clipped_lateral_accel

        # Convert acceleration to global
        global_acceleration = homo_rotate(acceleration, self.H_C2G)

        self.velocity += global_acceleration * dt

        self.position += self.velocity * dt

        return self.is_colliding()


    def is_colliding(self):

        rect = [
            [-self.width / 2, -self.height / 2],
            [self.width / 2, -self.height / 2],
            [self.width / 2, self.height / 2],
            [-self.width / 2, self.height / 2]
        ]
        
        world_rect = np.array([homo_apply(v, self.H_C2G) for v in rect]).astype(np.int32)

        minx = np.min(world_rect[:,0], axis=0)
        maxx = np.max(world_rect[:,0], axis=0)

        miny = np.min(world_rect[:,1], axis=0)
        maxy = np.max(world_rect[:,1], axis=0)

        roi = self.map.map_image[miny:maxy,minx:maxx]
        roi_rect = world_rect - np.array([minx, miny])
        mask = np.zeros((roi.shape[0], roi.shape[1], 1), np.uint8)

        cv2.fillConvexPoly(mask, roi_rect, (255, 255, 255))

        hit = cv2.bitwise_and(roi, roi, mask=mask)

        hit_wall = np.any((hit[:, :, 0] == 0) & (hit[:, :, 1] == 0) & (hit[:, :, 2] == 255))
        hit_finish = np.any((hit[:, :, 0] == 0) & (hit[:, :, 1] == 255) & (hit[:, :, 2] == 0))

        return ( hit_wall, hit_finish )


    def cast_ray(self, angle, step=2, max_distance=200):

        global_vec = homo_rotate( [ -np.sin(angle), np.cos(angle) ], self.H_C2G )

        point = np.array(self.position)

        for i in range(max_distance // step):
            point += global_vec*step

            hit_color = self.map.map_image[int(point[1]), int(point[0])]

            if ( hit_color[2] == 255 and np.sum(hit_color) == 255 ): 
                return Ray(angle, True, i*step, point)
            
        return Ray(angle, False, i*step, point)


    def render(self, image, static_image):
        # TODO: Make this draw an image instead

        rect = [
            [-self.width / 2, -self.height / 2],
            [self.width / 2, -self.height / 2],
            [self.width / 2, self.height / 2],
            [-self.width / 2, self.height / 2]
        ]
        
        world_rect = np.array([homo_apply(v, self.H_C2G) for v in rect]).astype(np.int32)

        if (self.drifting and self.last_world_rect is not None):
            for i in range(4):
                last_corner = self.last_world_rect[i]
                this_corner = world_rect[i]

                cv2.line(static_image, last_corner, this_corner, (95, 95, 95), 4)

        cv2.fillConvexPoly(image, world_rect, (0, 255, 0))

        self.last_world_rect = world_rect

        for ang in [ -np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2 ]:
            ray = self.cast_ray( ang )

            if (ray.did_hit):
                cv2.line(image, self.position.astype(np.int32), ray.hit_point.astype(np.int32), (50, 20, 220), 1, 16)
            else:
                cv2.line(image, self.position.astype(np.int32), ray.hit_point.astype(np.int32), (220, 20, 50), 1, 16)


    @property
    def forward_velocity(self):
        # Convert velocity to local
        local_velocity = homo_rotate(self.velocity, self.H_G2C)

        return local_velocity[1]

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
