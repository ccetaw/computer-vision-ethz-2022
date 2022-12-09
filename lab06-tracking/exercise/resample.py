import numpy as np

def resample(particles, particles_w):
    indexes = np.arange(particles.shape[0])
    result = np.random.choice(indexes, particles.shape[0], replace=True, p=particles_w)
    return particles[result, :], particles_w[result]