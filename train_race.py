import cv2
import numpy as np
from lib.Map import Map
from lib.Game import Game
from players.SimpleAI import SimpleAI
from players.SimpleAI import SimpleNetwork


map = Map("tracks/1_color.png", "tracks/1_map.png")
spawns = map.get_spawns()

try:
    vectors = np.load('tracks/1_reset_vectors.npy')
    map.reset_vectors = vectors
except Exception as e:
    print("No reset vectors found for map. Genarating them.")
    vectors = map.genarate_reset_vectors()
    np.save("tracks/1_reset_vectors.npy", vectors)


game = None
generation_time = 0

def mutate_network(network, mutation_rate=0.05):
    """Apply small random mutations to the weights of a network."""
    mutated_network = SimpleNetwork()  # Create a new network
    mutated_network.W1 = network.W1 + np.random.randn(*network.W1.shape) * mutation_rate
    mutated_network.b1 = network.b1 + np.random.randn(*network.b1.shape) * mutation_rate
    mutated_network.W2 = network.W2 + np.random.randn(*network.W2.shape) * mutation_rate
    mutated_network.b2 = network.b2 + np.random.randn(*network.b2.shape) * mutation_rate
    mutated_network.W3 = network.W3 + np.random.randn(*network.W3.shape) * mutation_rate
    mutated_network.b3 = network.b3 + np.random.randn(*network.b3.shape) * mutation_rate
    return mutated_network

def start_generation():
    global game, generation_time
    
    best_network = None
    best_performance = 0

    if generation_time > 0:
        for player in game.players:
            if player.performance > best_performance:
                best_performance = player.performance
                best_network = player.network
        
        best_network.save("best_network.npz")
        print("Performance: {:.4f}".format(best_performance / 1e5))
    else:
        best_network = SimpleNetwork()
        best_network.load("best_network.npz")

    game = Game(map)
    spawn_count = len(spawns)

    game.add_player(SimpleAI(map, (0, 255, 0), best_network))
    for i in range(spawn_count - 1):
        if (i % 2 == 0):
            mutated_network = mutate_network(best_network, mutation_rate=0.0005)
        else:
            mutated_network = mutate_network(best_network, mutation_rate=0.001)
        color = np.random.randint(0, 255, 3).tolist()

        # Pick one channel to be 255
        color[np.random.randint(0, 3)] = 255

        game.add_player(SimpleAI(map, color, mutated_network))

    generation_time = 0



cv2.namedWindow("Racecar", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Racecar", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

start_generation()

while True:

    generation_time += 1

    all_crashed = True
    lap_times = []

    for player in game.players:
        if len(player.lap_times) > 0:
            lap_times.append(player.lap_times[-1])

    for player in game.players:
        if not player.car.has_ever_crashed:
            all_crashed = False
            break
            

    if generation_time > 2600 or all_crashed:

        best_lap_time = min(lap_times) if (len(lap_times) > 0) else -1
        print(f"Best lap time: {best_lap_time}")
        start_generation()

    image = game.update(debug=False)

    text_width = cv2.getTextSize(f"Generation time: {generation_time}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
    cv2.putText(image, f"Generation time: {generation_time}", (1900 - text_width, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    sorted_players = sorted(game.players, key=lambda x: x.performance, reverse=True)
    for i, player in enumerate(sorted_players):
        text_width = cv2.getTextSize(f"Performance: {player.performance:.4f}", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
        cv2.putText(image, 'Performance: {:.4f}'.format(player.performance), (1900 - text_width, 60 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, player.car.color, 2)

    cv2.imshow("Racecar", image)

    key = cv2.waitKey(1)

    # Exit on escape
    if key == 27:
        break