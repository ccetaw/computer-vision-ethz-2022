import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    selected_frame = frame[ymin:ymax, xmin:xmax, :]

    r_hist, _ = np.histogram(selected_frame[:, :, 0], hist_bin)
    g_hist, _ = np.histogram(selected_frame[:, :, 1], hist_bin)
    b_hist, _ = np.histogram(selected_frame[:, :, 2], hist_bin)

    color_hist = np.array([r_hist, g_hist, b_hist])
    color_hist = color_hist / np.sum(color_hist)
    return color_hist