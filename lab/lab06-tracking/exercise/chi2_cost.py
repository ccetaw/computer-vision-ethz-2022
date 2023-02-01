import numpy as np


def chi2_cost(hist_x, hist):
    dist = np.sum( ((hist_x - hist) * (hist_x - hist)) / (hist_x + hist + 1e-8) )
    return dist