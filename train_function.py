import numpy as np
from players.SimpleAI import SimpleNetwork


def func(x):
    t = -1

    if (x[5] < 165.0/400.0 or x[2] > 250.0/400.0):
        t = 1
    c = x[0] + x[1] - x[3] - x[4]
    s = -c * 2.0

    return [t, s]

np.set_printoptions(suppress=True)

# Create 6xn matrix of random numbers
X = np.random.rand(100, 6)

# Apply the function to each row
Y = np.apply_along_axis(func, 1, X)

val = np.random.rand(6)
val_y = func(val)

net = SimpleNetwork()

for i in range(1000):
    avg_loss = 0

    for j in range(len(X)):
        x = X[j]
        y = Y[j]
        grads = net.backward(x, y)
        net.update_weights(grads, lr=0.008)

        avg_loss += np.mean(np.square(net.forward(x) - y))

    avg_loss /= len(X)

    if i % 100 == 0:
        # loss = np.mean(np.square(net.forward(X[0]) - Y[0]))
        print(f"Step {i}, Avg Loss: {avg_loss:.6f}")

print("Validation")
print(f"Input: {val}")
print(f"Target: {val_y}")
print(f"Prediction: {net.forward(val)}")

net.save("model.npz")