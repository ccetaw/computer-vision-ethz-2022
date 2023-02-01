import numpy as np

def estimate(particles, particles_w):
    particles_w = np.reshape(particles_w, (particles_w.shape[0], 1))
    return np.sum(particles*particles_w, axis=0)/np.sum(particles_w)