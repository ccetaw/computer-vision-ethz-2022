import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    selected_frame = frame[ymin:ymax, xmin:xmax, :]

    R_hist, _ = np.histogram(selected_frame[:, :, 0], hist_bin)
    G_hist, _ = np.histogram(selected_frame[:, :, 1], hist_bin)
    B_hist, _ = np.histogram(selected_frame[:, :, 2], hist_bin)

    result = np.array([R_hist, G_hist, B_hist])
    result = result / np.sum(result) #We are normalizing over all the values in the histogram
    return result