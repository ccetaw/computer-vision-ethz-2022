import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm

def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """
    vPoints = None  # numpy array, [nPointsX*nPointsY, 2]

    h = img.shape[0]+1
    w = img.shape[1]+1
    # Note that gird is not on pixel. (i,j) grid point is the bottom-right corner of (i,j) cell
    index_h = np.linspace(border, h-border, nPointsY, dtype=int)
    index_w = np.linspace(border, w-border, nPointsX, dtype=int)

    vy, vx = np.meshgrid(index_h, index_w)
    vx = vx.flatten()
    vy = vy.flatten()
    vPoints = np.array([vx, vy]).transpose()

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    # HOG created following the method in https://courses.cs.duke.edu/compsci527/fall15/notes/hog.pdf
    nBins = 8
    bin_len = 180.0/nBins
    w = cellWidth
    h = cellHeight

    grad_x = cv2.Sobel(img, cv2.CV_16S, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_16S, dx=0, dy=1, ksize=1)

    descriptors = []  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):
        center_x = round(vPoints[i, 0])
        center_y = round(vPoints[i, 1])

        desc = []
        for cell_y in range(-2, 2):
            for cell_x in range(-2, 2):
                start_y = center_y + (cell_y) * h
                end_y = center_y + (cell_y + 1) * h

                start_x = center_x + (cell_x) * w
                end_x = center_x + (cell_x + 1) * w

                # todo
                # compute the angles
                # print((grad_x[start_y:end_y, start_x:end_x]).shape)
                # print((grad_y[start_y:end_y, start_x:end_x]).shape)
                cell_grad_x = grad_x[start_y:end_y, start_x:end_x]
                cell_grad_y = grad_y[start_y:end_y, start_x:end_x]
            
                mags = np.sqrt(cell_grad_x**2+ cell_grad_y**2)
                angles = np.arctan2(cell_grad_y, cell_grad_x) * 180.0 /np.pi
                angles[angles < 0] += 180
                
                # mags, angles = cv2.cartToPolar(np.array(cell_grad_x, dtype=float), np.array(cell_grad_y, dtype=float), angleInDegrees=True)
                print("mags")
                print(mags)
                print("angles")
                print(angles)
                # compute the histogram
                j = np.int16(np.floor(angles / bin_len - 0.5))
                print("bin")
                print(j)
                v_j = mags * (((j+1) + 0.5) - angles/bin_len) 
                print("v_j")
                print(v_j)
                v_j1 = mags * (angles/bin_len - (j+0.5))
                j[j==-1] = nBins-1
                j[j==nBins] = 0
                print("v_j+1")
                print(v_j1)
                for bin in range(nBins):
                    desc.append(np.sum(v_j[j == bin]) + np.sum(v_j1[(j+1)%nBins == bin]))

        desc = np.asarray(desc) 
        desc /= np.linalg.norm(desc)
        descriptors.append(desc)

    descriptors = np.asarray(descriptors) # [nPointsX*nPointsY, 128], descriptor for the current image (100 grid points)
    return descriptors

if __name__ == '__main__':
    img = cv2.imread("./data/data_bow/cars-testing-neg/image_0051.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vPoints = grid_points(img, 10, 10, 8)
    descriptors = descriptors_hog(img, vPoints, 4, 4)
    print(descriptors.shape)