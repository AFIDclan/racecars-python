import cv2
import numpy as np
from Map import Map
from Game import Game
from Player import Player

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

game.add_player(Player(map, (0, 255, 0)))
game.add_player(Player(map, (255, 0, 0)))
# game.add_player(Player(map, (0, 0, 255)))
# game.add_player(Player(map, (255, 255, 0)))


cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:

    image = game.update(debug=False)

    cv2.imshow("Racecar", image)

    key = cv2.waitKey(1)

    # Exit on escape
    if key == 27:
        break