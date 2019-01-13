
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from calibration_functions import calibrateCamera_SLOW, undistort
from threshold_functions import binarize_image

def birdview(img, verbose=False):

    # Compute and apply perpective transform

    h, w = img.shape[:2]
    src = np.float32([[w, h-10],    
                      [0, h-10],    
                      [546, 460],   
                      [732, 460]])  
    dst = np.float32([[w, h],       
                      [0, h],       
                      [0, 0],       
                      [w, 0]])      

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_NEAREST)  # keep same size as input image

    if verbose:
        plt.figure()
        plt.imshow(warped, cmap='gray')
        plt.show()

    return warped, M, Minv

if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(calib_images_directory='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary, closing, opening = binarize_image(img_undistorted, verbose=False)

        img_birdview, M, Minv = birdview(opening, verbose=True)