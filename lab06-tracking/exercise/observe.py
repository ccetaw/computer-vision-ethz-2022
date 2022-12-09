import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    num_particles = particles.shape[0]
    result = np.zeros(num_particles)

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    for i in range(num_particles):
        
        xmin = round(particles[i, 0]-0.5*bbox_width)
        ymin = round(particles[i, 1]-0.5*bbox_height)
        xmax = round(particles[i, 0]+0.5*bbox_width)
        ymax = round(particles[i, 1]+0.5*bbox_height)

        xmin = min(max(0, xmin), frame_width-1)
        ymin = min(max(0, ymin), frame_height-1)
        xmax = min(max(0, xmax), frame_width-1)
        ymax = min(max(0, ymax), frame_height-1)
        sample_histo = color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin)

        distance = chi2_cost(sample_histo, hist)
        result[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-distance**2/(2*sigma_observe**2))
        
    result = result / np.sum(result)
    return result