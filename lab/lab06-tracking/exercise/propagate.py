import numpy as np

def propagate(particles, frame_height, frame_width, params):
    if params["model"] == 0:
        A = np.array([[1,0], [0,1]])
    if params["model"] == 1:
        A = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
    
    w = np.random.normal(loc=0, scale=params['sigma_position'], size=(2, particles.shape[0]))
    if params["model"] == 1:
        w_vel = np.random.normal(loc=0, scale=params['sigma_velocity'], size=(2, particles.shape[0]))
        w = np.vstack((w, w_vel))

    particles_n = np.transpose(A @ np.transpose(particles) + w)
    
    particles_n[:, 0] = np.minimum(particles_n[:, 0], frame_width -  1)
    particles_n[:, 1] = np.minimum(particles_n[:, 1], frame_height - 1)
    
    particles_n[:, 0] = np.maximum(particles_n[:, 0], 0)
    particles_n[:, 1] = np.maximum(particles_n[:, 1], 0)

    return particles_n