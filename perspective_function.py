import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

from calibration_functions import calibrateCamera_SLOW, undistort
from threshold_functions import binarize_image


def birdview(img, verbose=False):

    #img should be RGB Format
    # Compute and apply perpective transform

    h, w = img.shape[:2]
    # src = np.float32([[w, h-10],
    #                   [0, h-10],
    #                   [580, 477],
    #                   [699, 477]])
    # dst = np.float32([[w, h],
    #                   [0, h],
    #                   [0, 0],
    #                   [w, 0]])
    src = np.float32([[896., 675.],
                      [384., 675.],
                      [581, 460],
                      [699, 460]])
    dst = np.float32([[896., 720.],
                      [384., 720.],
                      [280., 0.],
                      [1024., 0.]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_NEAREST)

    if verbose:
        plt.figure()
        plt.imshow(warped, cmap='gray')
        plt.show()

    return warped, M, Minv


if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(
        calib_images_directory='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary, closing, opening = binarize_image(
            img_undistorted, verbose=False)

        img_birdview, M, Minv = birdview(img_binary, verbose=True)
