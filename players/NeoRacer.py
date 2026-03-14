import numpy as np
from lib.Player import Player


class NeoNetwork:
    """
    Feed-forward network: 11 inputs -> 32 -> 32 -> 2 outputs.
    ELU hidden activations, tanh output.
    Completely independent architecture — trained by train_neo.py.
    """

    N_IN  = 11   # 9 rays + forward_velocity + lateral_velocity
    N_H   = 32
    N_OUT = 2    # throttle, steer

    def __init__(self):
        # He initialisation (suits ELU)
        self.W1 = np.random.randn(self.N_H,  self.N_IN)  * np.sqrt(2.0 / self.N_IN)
        self.b1 = np.zeros(self.N_H)
        self.W2 = np.random.randn(self.N_H,  self.N_H)   * np.sqrt(2.0 / self.N_H)
        self.b2 = np.zeros(self.N_H)
        self.W3 = np.random.randn(self.N_OUT, self.N_H)  * np.sqrt(2.0 / self.N_H)
        self.b3 = np.zeros(self.N_OUT)

    # ── Parameter vector (used by ES trainer) ────────────────────────────────

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    def set_params(self, p: np.ndarray):
        idx = 0
        def take(n):
            nonlocal idx
            v = p[idx:idx + n]
            idx += n
            return v
        self.W1 = take(self.N_H  * self.N_IN ).reshape(self.N_H,  self.N_IN)
        self.b1 = take(self.N_H)
        self.W2 = take(self.N_H  * self.N_H  ).reshape(self.N_H,  self.N_H)
        self.b2 = take(self.N_H)
        self.W3 = take(self.N_OUT * self.N_H ).reshape(self.N_OUT, self.N_H)
        self.b3 = take(self.N_OUT)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def load(self, path: str):
        d = np.load(path)
        self.W1 = d['W1']; self.b1 = d['b1']
        self.W2 = d['W2']; self.b2 = d['b2']
        self.W3 = d['W3']; self.b3 = d['b3']

    # ── Inference ─────────────────────────────────────────────────────────────

    @staticmethod
    def _elu(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0.0, x, np.exp(x) - 1.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        h = self._elu(self.W1 @ x + self.b1)
        h = self._elu(self.W2 @ h + self.b2)
        return np.tanh(self.W3 @ h + self.b3)


class NeoRacer(Player):
    """
    Network-driven racer.
    9 front-arc rays (-90 to +90 degrees in 22.5-degree steps),
    plus normalised forward and lateral velocity as inputs.
    Train with:  python train_neo.py
    """

    _RAY_ANGLES = [i * np.pi / 8 - np.pi / 2 for i in range(9)]  # -90 .. +90 deg
    _RAY_MAX    = 400.0

    def __init__(self, map, color, network=None):
        super().__init__(map, color)
        if network is not None:
            self.network = network
        else:
            self.network = NeoNetwork()
            try:
                self.network.load("./players/neo_model.npz")
            except Exception:
                print("[NeoRacer] neo_model.npz not found — using random weights. Run train_neo.py first.")

        self.performance = 0.0

    def cast_rays(self):
        return [self.car.cast_ray(ang, 5, int(self._RAY_MAX))
                for ang in self._RAY_ANGLES]

    def control(self, rays):
        inputs = np.empty(NeoNetwork.N_IN, dtype=np.float64)
        for i, r in enumerate(rays):
            inputs[i] = r.distance / self._RAY_MAX
        inputs[9]  = self.car.forward_velocity / self.car.max_forward_velocity
        inputs[10] = self.car.lateral_velocity  / self.car.max_forward_velocity

        pred = self.network.forward(inputs)
        self.car.throttle = float(np.clip(pred[0], -1.0, 1.0))
        self.car.steer    = float(np.clip(pred[1], -1.0, 1.0))

        self.performance += max(0.0, self.car.forward_velocity)
