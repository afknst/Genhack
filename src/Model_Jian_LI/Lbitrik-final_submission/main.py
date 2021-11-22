import pickle
import numpy as np

from model import Model


def pload(_name):
    with open(f"{_name}.p", "rb") as _f:
        return pickle.load(_f)


THETA = pload('parameters')
NOISE = np.genfromtxt('noise.csv', delimiter=",")
G = Model(NOISE, THETA)

if __name__ == '__main__':
    SAMPLES = G.sample(size=408)
    np.savetxt("generated_samples.csv", SAMPLES, delimiter=",")
