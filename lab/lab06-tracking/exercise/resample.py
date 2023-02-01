import numpy as np

def resample(particles, particles_w):
    index = np.arange(particles.shape[0])
    sample = np.random.choice(index, particles.shape[0], replace=True, p=particles_w)
    return particles[sample, :], particles_w[sample]