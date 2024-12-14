from lib.Player import Player
import numpy as np

class GPT(Player):

    def cast_rays(self):
        # Cast more rays for better environmental awareness.
        # We'll cast rays across a wider arc to get a better idea of corners approaching.
        # Let's say we cast rays in front spread (-60°, -30°, 0°, 30°, 60°),
        # plus some side rays to sense walls directly to the left and right.
        angles = [
            -np.pi/3,    #  -60 degrees (front-left wide)
            -np.pi/6,    #  -30 degrees (front-left narrow)
            0,           #   0 degrees (straight ahead)
            np.pi/6,     #  30 degrees (front-right narrow)
            np.pi/3,     #  60 degrees (front-right wide)
            -np.pi/2,    #  -90 degrees (left)
            np.pi/2      #   90 degrees (right)
        ]
        return [ self.car.cast_ray(ang, 5, 400) for ang in angles ]

    def control(self, rays):
        # rays: 
        #   fl_wide, fl_narrow, f, fr_narrow, fr_wide, left_side, right_side
        fl_wide, fl_narrow, f, fr_narrow, fr_wide, left_side, right_side = rays

        # Extract distances
        fl_wide_dist    = fl_wide.distance
        fl_narrow_dist  = fl_narrow.distance
        f_dist          = f.distance
        fr_narrow_dist  = fr_narrow.distance
        fr_wide_dist    = fr_wide.distance
        left_dist       = left_side.distance
        right_dist      = right_side.distance

        # We'll try a more "smooth" control strategy:
        # 1. Adjust steering based on difference between left and right forward distances.
        #    We'll use primarily the "narrow" rays for immediate steering decision.
        #    The idea: if fr_narrow is longer than fl_narrow, there's more space on the right, so steer right.
        # 2. Use the wide rays to anticipate upcoming turns. If fl_wide < fr_wide, that suggests a left turn ahead, and vice versa.
        # 3. Reduce speed before corners by checking forward distance. If f_dist is small, slow down and turn more aggressively.

        # Compute some metrics:
        # Forward steering bias: difference between right and left forward space.
        forward_diff = (fr_narrow_dist - fl_narrow_dist)
        
        # Look ahead for corners using wide rays.
        # If fl_wide_dist is significantly smaller than fr_wide_dist, a turn to the left is needed.
        # If fr_wide_dist is smaller, a turn to the right is needed.
        corner_diff = (fr_wide_dist - fl_wide_dist)

        # Combine these two differences.
        # We'll make a weighted decision. The forward_diff helps immediate steering,
        # while corner_diff helps anticipating the correct direction further ahead.
        # A simple approach: steering angle = a blend of these differences.
        # Positive steering means turning right, negative means turning left.
        
        # Gain factors
        forward_gain = 0.002  # sensitivity to immediate differences
        corner_gain  = 0.0015 # sensitivity to upcoming corner differences

        # Base steering from these differences
        # If forward_diff > 0, it means more space on right front, so turn right.
        # If corner_diff > 0, it means more space on right side ahead, so maybe turn right to prepare.
        steer = forward_gain * forward_diff + corner_gain * corner_diff

        # Clip steering so it doesn't oscillate too wildly.
        # Steering range typically [-1, 1]. We'll keep within these.
        steer = max(min(steer, 1), -1)

        # If forward distance (f_dist) is small, we might be heading straight into a wall.
        # In that case, turn more aggressively towards the side that is more open and reduce speed.
        if f_dist < 150:
            # If we have very little forward space, we need to pick a direction quickly.
            # Choose the direction that has more space from the forward_diff.
            # Increase steer to turn away from the obstacle more aggressively.
            if forward_diff > 0:
                # turn right harder
                steer = 0.7
            else:
                # turn left harder
                steer = -0.7

        # Apply some damping if we're oscillating too much:
        # If we have a large discrepancy between left and right side distances (the side rays),
        # slow down and try to straighten out if possible to stabilize.
        # But this might not be strictly necessary. We'll rely on careful scaling of gains above.

        # Throttle control:
        # If front is clear (f_dist large), go faster.
        # If corner approaching (f_dist smaller), slow down to allow sharper turns.
        if f_dist > 300:
            throttle = 1.0
        elif f_dist > 200:
            throttle = 0.8
        elif f_dist > 100:
            throttle = 0.5
        else:
            throttle = 0.3  # slow down a lot when close to walls

        # To further reduce oscillation, if steering is large, also reduce throttle so we can turn more effectively.
        if abs(steer) > 0.5:
            throttle = min(throttle, 0.5)

        # Assign controls
        self.car.throttle = throttle
        self.car.steer = steer
