import numpy as np

def propagate(particles, frame_height, frame_width, params):
    if params["model"] == 0:
        A = np.asarray([[1,0], [0,1]])
    if params["model"] == 1:
        A = np.asarray([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]])
    
    w = np.random.normal(loc=0, scale=params['sigma_position'], size=(2, particles.shape[0]))
    if params["model"] == 1:
        w_vel = np.random.normal(loc=0, scale=params['sigma_velocity'], size=(2, particles.shape[0]))
        w = np.vstack((w, w_vel))

    result = np.transpose(np.matmul(A, np.transpose(particles)) + w)
    
    result[:, 0] = np.minimum(result[:, 0], frame_width - 1) #Era width
    result[:, 1] = np.minimum(result[:, 1], frame_height - 1) #Era height
    
    result[:, 0] = np.maximum(result[:, 0], 0)
    result[:, 1] = np.maximum(result[:, 1], 0)

    return result