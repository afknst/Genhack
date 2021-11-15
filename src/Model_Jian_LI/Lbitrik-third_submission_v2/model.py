from hashlib import sha256
import numpy as np
from scipy.stats import ks_2samp

D = 4


def norm(_m1, _m2):
    return (_m1 / 2 + _m2 / 20) / 2


def get_rng(Z):
    _sha = sha256(Z.tobytes())
    return int(_sha.hexdigest(), 16) % 2**23


class Model:
    # pylint: disable=R0902
    # PARAMETERS
    # theta: _CLF, _T, _X, _Y, _K
    # Z: np array
    def __init__(self, Z, theta):
        (
            self.CLF,
            self.T,
            self.X,
            self.Y,
            self.K,
        ) = theta
        self.EPS = np.min(self.T), np.max(self.T)

        self.SEED = get_rng(Z)
        self.RNG = np.random.default_rng(self.SEED)

        self.SIZE = len(self.T)
        _res = np.zeros(self.SIZE)
        for _i in range(self.SIZE):
            _res[_i] = np.count_nonzero(self.T < self.T[_i])
        self.Z = np.sort(_res)
        self.C = [np.sort(self.T[:, _]) for _ in range(D)]

    def get_seed(self):
        return self.RNG.integers(2**23)

    def N2P(self, _N):
        _ind = np.searchsorted(self.Y, _N)
        return (_N - self.Y[_ind - 1]) / self.K[_ind - 1] + self.X[_ind - 1]

    def p_value(self, a):
        def _ks(_a0, _a1):
            _res = ks_2samp(_a0, _a1)._asdict()
            _res['p'] = _res['pvalue']
            return _res

        p = []
        for i in range(D):
            res = _ks(self.T[:, i], a[:, i])
            p.append(res['p'])
        return np.mean(p)

    def score(self, a):
        _n = self.SIZE
        assert a.shape == (_n, D)

        def _Z(_a):
            _res = np.zeros(_n)
            for _i in range(_n):
                _res[_i] = np.count_nonzero(_a < _a[_i])
            return np.sort(_res)

        def _u(_d):
            _c0 = self.C[_d]
            _c1 = np.sort(a[:, _d])
            _res = np.zeros(_n)
            for _i in range(_n):
                _res[_i] = np.count_nonzero(_c0 <= _c1[_i]) + 1
            return _res / (_n + 2)

        def _W(_d):
            _ud = _u(_d)
            _c = np.arange(1, 2 * _n, 2)
            _res = _c * (np.log(_ud) + np.log((1 - np.flip(_ud))))
            return -_n - np.mean(_res)

        _m1 = np.mean([_W(_) for _ in range(D)])
        _m2 = np.mean(np.abs(self.Z - _Z(a)))

        return norm(_m1, _m2)

    def random_choice(self, a, size):
        _ind = self.RNG.integers(len(a), size=size)
        return a[_ind]

    def S1(self):
        self.CLF.set_params(random_state=self.get_seed())
        _G, _ = self.CLF.sample(3 * self.SIZE)
        _G = _G[~np.any(_G <= self.EPS[0], axis=1)]
        _G = _G[~np.any(_G >= self.EPS[1], axis=1)]
        return self.random_choice(_G, self.SIZE)

    def S1_better(self, _J0, steps=233):
        for _ in range(steps):
            _G1 = self.S1()
            _J1 = self.score(_G1)
            if _J1 < _J0:
                return _G1, _J1
        return None

    def MC(self, _th=7):
        _G0 = self.S1()
        _J0 = self.score(_G0)
        _i = 0
        while True:
            print(_J0)
            _NEW = self.S1_better(_J0)
            if _NEW is not None:
                _G0, _J0 = _NEW
                _i = 0
            else:
                _i += 1

            if _i >= _th:
                break
        return _G0

    def sample(self, size=410):
        _G = self.MC()
        while size > len(_G):
            _G = np.concatenate([_G, self.MC()])

        _SAMPLES = self.random_choice(_G, size)
        _p = self.p_value(_SAMPLES)

        for _ in range(size):
            print(_p)
            _SAMPLES_NEW = self.random_choice(_G, size)
            _p_NEW = self.p_value(_SAMPLES_NEW)
            if _p_NEW > _p:
                _SAMPLES = _SAMPLES_NEW
                _p = _p_NEW

        return self.N2P(_SAMPLES)
