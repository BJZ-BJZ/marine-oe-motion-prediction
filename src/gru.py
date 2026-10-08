"""NumPy GRU extracted from the archived supplementary experiment."""
import math
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

class NumpyGRU:
    def __init__(self, input_size: int, context_size: int, hidden: int, output_size: int, seed: int):
        rng = np.random.default_rng(seed)
        def weight(a, b):
            limit = math.sqrt(6.0 / (a + b))
            return rng.uniform(-limit, limit, size=(a, b)).astype(np.float64)
        self.p = {
            "Wz": weight(input_size, hidden), "Uz": weight(hidden, hidden), "bz": np.zeros(hidden),
            "Wr": weight(input_size, hidden), "Ur": weight(hidden, hidden), "br": np.zeros(hidden),
            "Wn": weight(input_size, hidden), "Un": weight(hidden, hidden), "bn": np.zeros(hidden),
            "Wo": weight(hidden + context_size, output_size), "bo": np.zeros(output_size),
        }

    def forward(self, seq: np.ndarray, context: np.ndarray, cache: bool = False):
        h = np.zeros((len(seq), self.p["Uz"].shape[0]), dtype=np.float64)
        steps = []
        for t in range(seq.shape[1]):
            x = seq[:, t, :]
            hp = h
            z = sigmoid(x @ self.p["Wz"] + hp @ self.p["Uz"] + self.p["bz"])
            r = sigmoid(x @ self.p["Wr"] + hp @ self.p["Ur"] + self.p["br"])
            n = np.tanh(x @ self.p["Wn"] + (r * hp) @ self.p["Un"] + self.p["bn"])
            h = (1.0 - z) * n + z * hp
            if cache:
                steps.append((x, hp, z, r, n))
        joined = np.concatenate([h, context], axis=1)
        y = joined @ self.p["Wo"] + self.p["bo"]
        return (y, (steps, joined)) if cache else y

    def loss_and_grad(self, seq: np.ndarray, context: np.ndarray, truth: np.ndarray):
        pred, (steps, joined) = self.forward(seq, context, cache=True)
        residual = pred - truth
        loss = float(np.mean(residual * residual))
        dy = (2.0 / residual.size) * residual
        g = {name: np.zeros_like(value) for name, value in self.p.items()}
        g["Wo"] = joined.T @ dy
        g["bo"] = dy.sum(axis=0)
        dh = dy @ self.p["Wo"][: self.p["Uz"].shape[0], :].T
        for x, hp, z, r, n in reversed(steps):
            dn = dh * (1.0 - z)
            dz = dh * (hp - n)
            dhp = dh * z
            dan = dn * (1.0 - n * n)
            g["Wn"] += x.T @ dan
            g["Un"] += (r * hp).T @ dan
            g["bn"] += dan.sum(axis=0)
            drhp = dan @ self.p["Un"].T
            dr = drhp * hp
            dhp += drhp * r
            dar = dr * r * (1.0 - r)
            g["Wr"] += x.T @ dar
            g["Ur"] += hp.T @ dar
            g["br"] += dar.sum(axis=0)
            dhp += dar @ self.p["Ur"].T
            daz = dz * z * (1.0 - z)
            g["Wz"] += x.T @ daz
            g["Uz"] += hp.T @ daz
            g["bz"] += daz.sum(axis=0)
            dhp += daz @ self.p["Uz"].T
            dh = dhp
        return loss, g

    def state(self):
        return {name: value.copy() for name, value in self.p.items()}

    def load_state(self, state):
        self.p = {name: value.copy() for name, value in state.items()}
