import json
import pickle

import numpy as np


class NpEncoder(json.JSONEncoder):
    # pylint: disable=W0221
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save(_A, _name):
    with open(f'{_name}.npy', 'wb') as _f:
        np.save(_f, np.array(_A))


def load(_name):
    with open(f'{_name}.npy', 'rb') as _f:
        return np.load(_f)


def psave(_A, _name):
    with open(f"{_name}.p", "wb") as _f:
        pickle.dump(_A, _f)


def pload(_name):
    with open(f"{_name}.p", "rb") as _f:
        return pickle.load(_f)


def jsave(_dict, _name):
    with open(f"{_name}.json", 'w') as _f:
        json.dump(_dict, _f, indent=4, ensure_ascii=False, cls=NpEncoder)


def jload(_name):
    with open(f"{_name}.json", 'r') as _f:
        return json.load(_f)
