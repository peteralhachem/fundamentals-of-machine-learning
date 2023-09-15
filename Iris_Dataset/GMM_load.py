import json
import numpy


def save_gmm(gmm, filename):
    gmm_json = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmm_json, f)


def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, numpy.asarray(j), numpy.asarray(k)) for i, j, k in gmm]
