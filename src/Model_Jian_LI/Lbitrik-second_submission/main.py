import pickle
import numpy as np


def pload(_name):
    with open(f"{_name}.p", "rb") as _f:
        return pickle.load(_f)


D = 4
CLF, X, Y, K, MU, EPS, RNG = pload('parameters')
N_INTERP = len(Y)
X_LAST = X[-1]

NOISE = MU * np.genfromtxt('noise.csv', delimiter=",")
GUILD = "Lbitrik"


def N2P(_N):
    _ind = np.searchsorted(Y, _N)
    _P = (_N - Y[_ind - 1]) / K[_ind - 1] + X[_ind - 1]
    _P[_ind == 0] = 0
    _P[_ind == N_INTERP + 1] = X_LAST
    return _P


def samples(clf=CLF, eps=EPS, size=410, rng=RNG):
    clf.set_params(random_state=rng)
    rng = np.random.default_rng(rng)

    _G, _ = clf.sample(3 * size)
    rng.normal(size=(3 * size, D))
    _G += NOISE
    _G = _G[~np.any(_G <= eps[0], axis=1)]
    _G = _G[~np.any(_G >= eps[1], axis=1)]
    _ind = rng.integers(len(_G), size=size)
    return N2P(_G[_ind])


if __name__ == '__main__':
    SAMPLES = samples()
    np.savetxt(f"{GUILD}-second_submission.csv", SAMPLES, delimiter=",")
