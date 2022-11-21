import time
import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

# Compute the distance between a given point and all other points
# that are within a specific radius. In this assignment, the radius is infinity
def distance(x, X):
    dist = (X-x).norm(dim=1)
    return dist

def distance_batch(X):
    p = torch.norm(X, dim=1) * torch.ones(X.shape[0], X.shape[0])
    q = (torch.norm(X, dim=1) * torch.ones(X.shape[0], X.shape[0])).transpose(-1,0)
    pq = X.matmul(X.transpose(-1,0))
    # print(p**2)
    # print(q**2)
    # print(pq)
    return torch.sqrt(torch.abs(p**2 - 2*pq + q**2))

def gaussian(dist, bandwidth):
    # Without normalization
    weight = torch.exp(-0.5 * (dist/bandwidth)**2)

    # With normalization
    # weight = 1/np.sqrt(2*np.pi)/bandwidth * torch.exp(-0.5 * (dist/bandwidth)**2)

    return weight

def update_point(weight, X):
    # x = torch.zeros(X.shape[1])
    # for i, y in enumerate(X):
    #     x += weight[i]*y

    # return x
    return weight.matmul(X) / weight.sum()

def update_point_batch(weight, X):
    return weight.matmul(X) / weight.sum(dim=1, keepdim=True)

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    dist = distance_batch(X)
    weight = gaussian(dist, bandwidth)
    X_ = update_point_batch(weight, X)
    return X_


def meanshift(X):
    X = X.clone()
    for _ in tqdm(range(20)):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
# print(image.shape)
image_lab = color.rgb2lab(image)
# print(image_lab.shape)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image
# print(image_lab.shape)

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
if centroids.shape[0] > colors.shape[0]:
    print("Current modes found: {}".format(centroids.shape[0]))
    raise IndexError("Not converged!")

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
