import cv2
import numpy as np
from lib.Map import Map
from lib.Game import Game
from lib.Player import Player

from players.BangBang import BangBang
from players.SimpleAI import SimpleAI, SimpleNetwork


map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

try:
    vectors = np.load('tracks/1_reset_vectors.npy')
    map.reset_vectors = vectors
except Exception as e:
    print("No reset vectors found for map. Genarating them.")
    vectors = map.genarate_reset_vectors()
    np.save("tracks/1_reset_vectors.npy", vectors)


game = Game(map)

class TrainedAI(SimpleAI):
    def __init__(self, map, color):
        super().__init__(map, color)
        self.network = SimpleNetwork()
        self.network.load("./best_network.npz")

game.add_player(SimpleAI(map, (0, 255, 0)))
game.add_player(BangBang(map, (255, 0, 0)))
game.add_player(TrainedAI(map, (0, 0, 255)))


cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:

    image = game.update(debug=False)

    cv2.imshow("Racecar", image)

    key = cv2.waitKey(1)

    # Exit on escape
    if key == 27:
        break