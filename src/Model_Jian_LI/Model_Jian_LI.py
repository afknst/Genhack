import pickle
import numpy as np


def pload(_name):
    with open(f"{_name}.p", "rb") as _f:
        return pickle.load(_f)


D = 4
SIG, CLF, MU, N, RNG = pload('parameters')
SIG = np.array(SIG)
NOISE = MU * np.genfromtxt('noise.csv', delimiter=",")
GUILD = "Lbitrik"


def samples(clf=CLF, size=N, rng=RNG):
    clf.set_params(random_state=rng)
    _G, _ = clf.sample(size)
    _G += NOISE
    _G = np.maximum(_G, 1e-6)
    _G = np.minimum(_G, 1 - 1e-6)
    return np.sqrt(-2 * np.log(_G)) * SIG


if __name__ == '__main__':
    SAMPLES = samples()
    np.savetxt(f"{GUILD}-first_submission.csv", SAMPLES, delimiter=",")
