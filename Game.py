import time
import numpy as np

class Game:
    def __init__(self, map):
        self.map = map
        self.players = []
        self.last_update = time.time()

        self.unused_spawns = self.map.get_spawns()

    def add_player(self, player):

        try:
            player.car.position = np.array(self.unused_spawns.pop(0)).astype(np.float32)
            player.car.angle = -np.pi / 2.0

            self.players.append(player)
        except Exception as e:
            print("No more spawns left")
            print(e)
            return


        self.players.append(player)

    def update(self, debug=True):

        delta = (time.time() - self.last_update)
        self.last_update = time.time()

        display_image = self.map.color_image.copy()   

        for player in self.players:

            rays = player.cast_rays()
            player.control(rays)

            hit_wall, hit_finish = player.car.update(delta)

            if (hit_wall):
                self.map.reset_car(player.car)

            if (hit_finish and player.cleared_finish):
                player.lap_times.append(time.time() - player.current_lap_start)
                player.current_lap_start = time.time()
                player.cleared_finish = False

            # Set player as elegible for crossing the finish if they go on the top half of the screen
            if (player.car.position[1] < 640 and not player.cleared_finish):
                player.cleared_finish = True

            if (debug):
                player.car.render(display_image, self.map.color_image, rays)
            else:
                player.car.render(display_image, self.map.color_image)
        
        return display_image