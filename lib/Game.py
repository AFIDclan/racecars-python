import time
import numpy as np
import cv2

class Game:
    def __init__(self, map):
        self.map = map
        self.players = []
        self.last_update = time.time()

        self.unused_spawns = self.map.get_spawns()

        self.tick_time_seconds = 0.033
        self.ticks = 0

    def add_player(self, player):

        try:
            player.car.position = np.array(self.unused_spawns.pop(0)).astype(np.float32)
            player.car.angle = -np.pi / 2.0

            self.players.append(player)
        except Exception as e:
            print("No more spawns left")
            print(e)
            return


    def update(self, debug=True):

        self.ticks += 1

        self.last_update = time.time()

        display_image = self.map.color_image.copy()   

        all_lap_times = []

        for player in self.players:

            rays = player.cast_rays()
            player.control(rays)

            hit_wall, hit_finish = player.car.update(self.tick_time_seconds)

            if (hit_wall):
                self.map.reset_car(player.car)

            if (hit_finish and player.cleared_finish):
                player.lap_times.append((self.ticks - player.current_lap_start) * self.tick_time_seconds)
                player.current_lap_start = self.ticks
                player.cleared_finish = False

            # Set player as elegible for crossing the finish if they go on the top half of the screen
            if (player.car.position[1] < 640 and not player.cleared_finish):
                player.cleared_finish = True

            if (debug):
                player.car.render(display_image, self.map.color_image, rays)
            else:
                player.car.render(display_image, self.map.color_image)

            # Add constructor name text above the car
            text_width, text_height = cv2.getTextSize(player.__class__.__name__, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.putText(display_image, player.__class__.__name__, (int(player.car.position[0] - text_width/2), int(player.car.position[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.car.color, 2)

            for times in player.lap_times:
                all_lap_times.append([player, times])

        # Sort the lap times
        all_lap_times.sort(key=lambda x: x[1])

        # Draw the lap times
        for i, (player, times) in enumerate(all_lap_times):
            cv2.putText(display_image, f"{i+1} - {times:.2f}", (10, 30 + 20*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, player.car.color, 2)
        
        return display_image