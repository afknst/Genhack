import pickle
from hashlib import sha256
import numpy as np


def pload(_name):
    with open(f"{_name}.p", "rb") as _f:
        return pickle.load(_f)


def get_rng(noise):
    _sha = sha256(noise.tobytes())
    return int(_sha.hexdigest(), 16) % 2**23


class Model:
    def __init__(self, _CLF, _X, _Y, _K, _EPS):
        self.CLF = _CLF
        self.X = _X
        self.Y = _Y
        self.K = _K
        self.EPS = _EPS

    def N2P(self, _N):
        _ind = np.searchsorted(self.Y, _N)
        return (_N - self.Y[_ind - 1]) / self.K[_ind - 1] + self.X[_ind - 1]

    def samples(self, noise, size=410):
        _rng = get_rng(noise)
        self.CLF.set_params(random_state=_rng)
        rng = np.random.default_rng(_rng)
        _G, _ = self.CLF.sample(3 * size)
        _G = _G[~np.any(_G <= self.EPS[0], axis=1)]
        _G = _G[~np.any(_G >= self.EPS[1], axis=1)]
        _ind = rng.integers(len(_G), size=size)
        return self.N2P(_G[_ind])


CLF, X, Y, K, EPS = pload('parameters')
G = Model(CLF, X, Y, K, EPS)
NOISE = np.genfromtxt('noise.csv', delimiter=",")

if __name__ == '__main__':
    SAMPLES = G.samples(NOISE, size=410)
    np.savetxt("generated_samples.csv", SAMPLES, delimiter=",")
