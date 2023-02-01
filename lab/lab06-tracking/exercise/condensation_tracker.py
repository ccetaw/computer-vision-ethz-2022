import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib import patches

from color_histogram import color_histogram
from propagate import propagate
from observe import observe
from resample import resample
from estimate import estimate


top_left = []
bottom_right = []

def line_select_callback(clk, rls):
    print(clk.xdata, clk.ydata)
    global top_left
    global bottom_right
    top_left = (int(clk.xdata), int(clk.ydata))
    bottom_right = (int(rls.xdata), int(rls.ydata))


def onkeypress(event):
    global top_left
    global bottom_right
    global img
    if event.key == 'q':
        print('final bbox', top_left, bottom_right)
        plt.close()


def toggle_selector(event):
    toggle_selector.RS.set_active(True)


def condensation_tracker(video_name, params):
    '''
    video_name - video name
    params - parameters
        - draw_plats {0, 1} draw output plots throughout
        - hist_bin   1-255 number of histogram bins for each color: proper values 4,8,16
        - alpha      number in [0,1]; color histogram update parameter (0 = no update)
        - sigma_position   std. dev. of system model position noise
        - sigma_observe    std. dev. of observation model noise
        - num_particles    number of particles
        - model            {0,1} system model (0 = no motion, 1 = constant velocity)
    if using model = 1 then the following parameters are used:
        - sigma_velocity   std. dev. of system model velocity noise
        - initial_velocity initial velocity to set particles to
    '''
    # Choose video
    if video_name == "video1.avi":
        first_frame = 10
        last_frame = 42
    elif video_name == "video2.avi":
        first_frame = 3
        last_frame = 38
    elif video_name == "video3.avi":
        first_frame = 1
        last_frame = 60

    # Change this to where your data is
    data_dir = '../data/'
    video_path = os.path.join(data_dir, video_name)

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(1, first_frame)
    ret, first_image = vidcap.read()

    fig, ax = plt.subplots(1)
    image = first_image
    frame_height = first_image.shape[0]
    frame_width = first_image.shape[1]

    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    ax.imshow(first_image)

    toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
    bbox = plt.connect('key_press_event', toggle_selector)
    key = plt.connect('key_press_event', onkeypress)
    plt.title("Draw a box then press 'q' to continue")
    plt.show()

    bbox_width = bottom_right[0] - top_left[0]
    bbox_height = bottom_right[1] - top_left[1]

    # Get initial color histogram
    # === implement fuction color_histogram() ===
    hist = color_histogram(top_left[0], top_left[1], bottom_right[0], bottom_right[1],
                first_image, params["hist_bin"])
    # ===========================================

    state_length = 2
    if(params["model"] == 1):
        state_length = 4

    # a priori mean state
    mean_state_a_priori = np.zeros([last_frame - first_frame + 1, state_length])
    mean_state_a_posteriori = np.zeros([last_frame - first_frame + 1, state_length])
    # bounding box centre
    mean_state_a_priori[0, 0:2] = [(top_left[0] + bottom_right[0])/2., (top_left[1] + bottom_right[1])/2.]

    if params["model"] == 1:
        # use initial velocity
        mean_state_a_priori[0, 2:4] = params["initial_velocity"]

    # Initialize Particles
    particles = np.tile(mean_state_a_priori[0], (params["num_particles"], 1))
    particles_w = np.ones([params["num_particles"], 1]) * 1./params["num_particles"]

    fig, ax = plt.subplots(1)
    im = ax.imshow(first_image)
    plt.ion()

    for i in range(last_frame - first_frame + 1):
        t = i + first_frame

        # Propagate particles
        # === Implement function propagate() ===
        particles = propagate(particles, frame_height, frame_width, params)
        # ======================================

        # Estimate
        # === Implement function estimate() ===
        mean_state_a_priori[i, :] = estimate(particles, particles_w)
        # ======================================

        # Get frame
        ret, frame = vidcap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw
        if params["draw_plots"]:
            ax.set_title("Frame: %d" % t)
            im.set_data(frame)
            to_remove = []

            # Plot a priori particles
            new_plot = ax.scatter(particles[:, 0], particles[:, 1], color='blue', s=2)
            to_remove.append(new_plot)

            # Plot a priori estimation
            for j in range(i-1, -1, -1):
                lwidth = 30 - 3 * (i-j)
                if lwidth > 0:
                    new_plot = ax.scatter(mean_state_a_priori[j+1, 0], mean_state_a_priori[j+1, 1], color='blue', s=lwidth)
                    to_remove.append(new_plot)
                if j != i:
                    new_plot = ax.plot([mean_state_a_priori[j, 0], mean_state_a_priori[j+1, 0]], 
                                       [mean_state_a_priori[j, 1], mean_state_a_priori[j+1, 1]], color='blue')
                    to_remove.append(new_plot[0])

            # Plot a priori bounding box
            if not np.any(np.isnan(mean_state_a_priori[i, :])):
                patch = ax.add_patch(patches.Rectangle((mean_state_a_priori[i, 0] - 0.5 * bbox_width, mean_state_a_priori[i, 1] - 0.5 * bbox_height),
                                                        bbox_width, bbox_height, fill=False, edgecolor='blue', lw=2))
                to_remove.append(patch)

        # Observe
        # === Implement function observe() ===
        particles_w = observe(particles, frame, bbox_height, bbox_width, params["hist_bin"], hist, params["sigma_observe"])
        # ====================================

        # Update estimation
        mean_state_a_posteriori[i, :] = estimate(particles, particles_w)

        # Update histogram color model                   
        hist_crrent = color_histogram(min(max(0, round(mean_state_a_posteriori[i, 0]-0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(mean_state_a_posteriori[i, 1]-0.5*bbox_height)), frame_height-1),
                                      min(max(0, round(mean_state_a_posteriori[i, 0]+0.5*bbox_width)), frame_width-1),
                                      min(max(0, round(mean_state_a_posteriori[i, 1]+0.5*bbox_height)), frame_height-1),
                                      frame, params["hist_bin"])

        hist = (1 - params["alpha"]) * hist + params["alpha"] * hist_crrent

        if params["draw_plots"]:
            # Plot updated estimation
            for j in range(i-1, -1, -1):
                lwidth = 30 - 3 * (i-j)
                if lwidth > 0:
                    new_plot = ax.scatter(mean_state_a_posteriori[j+1, 0], mean_state_a_posteriori[j+1, 1], color='red', s=lwidth)
                    to_remove.append(new_plot)
                if j != i:
                    new_plot = ax.plot([mean_state_a_posteriori[j, 0], mean_state_a_posteriori[j+1, 0]], 
                                       [mean_state_a_posteriori[j, 1], mean_state_a_posteriori[j+1, 1]], color='red')
                    to_remove.append(new_plot[0])
            
            # Plot updated bounding box
            if not np.any(np.isnan(mean_state_a_posteriori[i, :])):
                patch = ax.add_patch(patches.Rectangle((mean_state_a_posteriori[i, 0] - 0.5 * bbox_width, mean_state_a_posteriori[i, 1] - 0.5 * bbox_height),
                                                        bbox_width, bbox_height, fill=False, edgecolor='red', lw=2))
                to_remove.append(patch)


        # RESAMPLE PARTICLES
        # === Implement function resample() ===
        particles, particles_w = resample(particles, particles_w)
        # =====================================

        if params["draw_plots"] and t != last_frame:
            
            plt.pause(0.2)
            # Remove previous element from plot
            for e in to_remove:
                e.remove()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    video_name = 'video3.avi'
    params = {
        "draw_plots": 1,
        "hist_bin": 16,
        "alpha": 0,
        "sigma_observe": 0.1,
        "model": 1,
        "num_particles": 300,
        "sigma_position": 15,
        "sigma_velocity": 5,
        "initial_velocity": (10, 0)
    }
    condensation_tracker(video_name, params)
