
import numpy as np
import cv2
import glob
import os.path as path
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg


#chessboard size 
m = 9
n = 6

def calibrateCamera_decorater(func):
    """
    Decorater for the calibrateCamera_SLOW() to avoid the calculation of the calibration parameters everytime we run the program.
    """

    calibration_data = 'camera_cal/calibration_data.pickle'

    def timesaver(*args, **kwargs):
        if path.exists(calibration_data):
                print('Loading calculated camera calibration...', end=' ')
                with open(calibration_data, 'rb') as dump_file:
                    calibration_param = pickle.load(dump_file)
        else:
            print('Computing camera calibration_param...', end=' ')
            calibration_param = func(*args, **kwargs)
            with open(calibration_data, 'wb') as dump_file:
                pickle.dump(calibration_param, dump_file)
        print('Done.')
        return calibration_param

    return timesaver


@calibrateCamera_decorater
def calibrateCamera_SLOW(calib_images_directory, verbose=False):

    
    assert path.exists(calib_images_directory), '"{}" The folder must exist and contain calibration images.'.format(calib_images_directory)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:m, 0:n].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path.join(calib_images_directory, 'calibration*.jpg'))

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (m,n), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            if verbose:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (m,n), corners, ret)
                #write_name = 'corners_found'+str(idx)+'.jpg'
                #cv2.imwrite(write_name, img)
                cv2.imshow('img', img)
                cv2.waitKey(500)
    if verbose:
        cv2.destroyAllWindows()
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1],None,None)

    return ret, mtx, dist, rvecs, tvecs

def undistort(img, mtx, dist, verbose=False):
    """
    Undistort the image givine the camera's distortion and matrix coefficients
    :param image: input image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param verbose: if True, show frame before/after distortion correction
    :return: undistorted image
    """
    image_undistorted = cv2.undistort(img, mtx, dist, None, newCameraMatrix=mtx)

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return image_undistorted

if __name__ == '__main__':

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(calib_images_directory='camera_cal', verbose=False)

    img = cv2.imread('test_images/test2.jpg')

    img_undistorted = undistort(img, mtx, dist, verbose=True)

    cv2.imwrite('calibration_output/test2_before_calibration.jpg', img)
    cv2.imwrite('calibration_output/test2_after_calibration.jpg', img_undistorted)



