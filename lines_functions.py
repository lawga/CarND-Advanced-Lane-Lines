import collections
import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from calibration_functions import calibrateCamera_SLOW, undistort
from globals import xm_per_pix, ym_per_pix
from perspective_function import birdview
from threshold_functions import binarize_image

# Define a class to receive the characteristics of each line detection


class Line():

    def __init__(self, buffer_length=10):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []

        # polynomial coefficients for the most recent fit
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=2 * buffer_length)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_length)

        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def draw(self, mask, color=(0, 255, 0), line_width=50, average=False):
        """
        Draw the line on a color mask image.
        """
        h, w, c = mask.shape

        plot_y = np.linspace(0, h - 1, h)
        coeffs = self.average_fit if average else self.last_fit_pixel

        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(list(zip(line_left_side, plot_y)))
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))
        pts = np.hstack([pts_left, pts_right])

        # Draw the lane onto the warped blank image
        return cv2.fillPoly(mask, [np.int32(pts)], color)

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """
        Update Line with new fitted coefficients.

        :param new_fit_pixel: new polynomial coefficients (pixel)
        :param new_fit_meter: new polynomial coefficients (meter)
        :param detected: if the Line was detected or inferred
        :param clear_buffer: if True, reset state
        :return: None
        """
        self.detected = detected

        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    # PROPERTIES

    @property
    # polynomial coefficients averaged over the last n iterations
    def best_fit_pixels(self):
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    # polynomial coefficients averaged over the last n iterations
    def best_fit_meters(self):
        return np.mean(self.recent_fits_meter, axis=0)

    @property
    # radius of curvature of the line in some units
    def radius_of_curvature_pixels(self):
        y_eval = 0
        coeffs = self.best_fit_pixels
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    # radius of curvature of the line in some units
    def radius_of_curvature_meters(self):
        y_eval = 0
        coeffs = self.last_fit_meter
        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])


def find_lane_pixels(binary_warped, line_L, line_R, nwindows=9, verbose=False):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS

    """ 
    # Choose the number of sliding windows
    nwindows = 9 """
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        out_img = cv2.rectangle(
            out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        out_img = cv2.rectangle(
            out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                          (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                           (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions

    # leftx = nonzerox[left_lane_inds]
    # lefty = nonzeroy[left_lane_inds]
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds]

    leftx = line_L.allx = nonzerox[left_lane_inds]
    lefty = line_L.ally = nonzeroy[left_lane_inds]
    rightx = line_R.allx = nonzerox[right_lane_inds]
    righty = line_R.ally = nonzeroy[right_lane_inds]

    detected = True
    if not list(line_L.allx) or not list(line_L.ally):
        left_fit_pixel = line_L.last_fit_pixel
        left_fit_meter = line_L.last_fit_meter
        detected = False
    else:
        left_fit_pixel = np.polyfit(line_L.ally, line_L.allx, 2)
        left_fit_meter = np.polyfit(
            line_L.ally * ym_per_pix, line_L.allx * xm_per_pix, 2)

    if not list(line_R.allx) or not list(line_R.ally):
        right_fit_pixel = line_R.last_fit_pixel
        right_fit_meter = line_R.last_fit_meter
        detected = False
    else:
        right_fit_pixel = np.polyfit(line_R.ally, line_R.allx, 2)
        right_fit_meter = np.polyfit(
            line_R.ally * ym_per_pix, line_R.allx * xm_per_pix, 2)

    

    #Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
                                                                                                                                                                                                    
    left_fit_meter = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_meter = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)


    line_L.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_R.update_line(right_fit_pixel, right_fit_meter, detected=detected)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if verbose:
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(out_img, cmap='gray')
        figManager = plt.get_current_fig_manager()  # to control the figure to be showen
        # maximaize the window of the plot to cover the whole screen
        figManager.window.showMaximized()
        plt.show()

    return line_L, line_R, out_img

def get_fits_by_previous_fits(birdeye_binary, line_L, line_R, verbose=False):
    """
    Get polynomial coefficients for lane-lines detected in an binary image.
    This function starts from previously detected lane-lines to speed-up the search of lane-lines in the current frame.

    :param birdeye_binary: input bird's eye view binary image
    :param line_L: left lane-line previously detected
    :param line_R: left lane-line previously detected
    :param verbose: if True, display intermediate output
    :return: updated lane lines and output image
    """

    height, width = birdeye_binary.shape

    left_fit_pixel = line_L.last_fit_pixel
    right_fit_pixel = line_R.last_fit_pixel

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = birdeye_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set the width of the windows +/- margin
    margin = 100

    # Identify the nonzero pixels in x and y within the previous detected line-lane
    left_lane_inds = (
    (nonzerox > (left_fit_pixel[0] * (nonzeroy ** 2) + left_fit_pixel[1] * nonzeroy + left_fit_pixel[2] - margin)) & (
    nonzerox < (left_fit_pixel[0] * (nonzeroy ** 2) + left_fit_pixel[1] * nonzeroy + left_fit_pixel[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit_pixel[0] * (nonzeroy ** 2) + right_fit_pixel[1] * nonzeroy + right_fit_pixel[2] - margin)) & (
    nonzerox < (right_fit_pixel[0] * (nonzeroy ** 2) + right_fit_pixel[1] * nonzeroy + right_fit_pixel[2] + margin)))

    # Extract left and right line pixel positions
    line_L.allx, line_L.ally = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    line_R.allx, line_R.ally = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # check if lane-line are detected in the prefious frame, if so, then load the fitting coefficents from the last frame. if not, then 
    detected = True
    if not list(line_L.allx) or not list(line_L.ally):
        # left_fit_pixel = line_L.best_fit_pixel
        # left_fit_meter = line_L.best_fit_meter
        left_fit_pixel = line_L.last_fit_pixel
        left_fit_meter = line_L.last_fit_meter
        detected = False
    else:
        # left_fit_pixel = line_L.best_fit_pixels
        # left_fit_meter = line_L.best_fit_meters
        left_fit_pixel = np.polyfit(line_L.ally, line_L.allx, 2)
        left_fit_meter = np.polyfit(line_L.ally * ym_per_pix, line_L.allx * xm_per_pix, 2)

    if not list(line_R.allx) or not list(line_R.ally):
        # right_fit_pixel = line_R.best_fit_pixel
        # right_fit_meter = line_R.best_fit_meter
        right_fit_pixel = line_R.last_fit_pixel
        right_fit_meter = line_R.last_fit_meter
        detected = False
    else:
        # right_fit_pixel = line_R.best_fit_pixels
        # right_fit_meter = line_R.best_fit_meters
        right_fit_pixel = np.polyfit(line_R.ally, line_R.allx, 2)
        right_fit_meter = np.polyfit(line_R.ally * ym_per_pix, line_R.allx * xm_per_pix, 2)

    line_L.update_line(left_fit_pixel, left_fit_meter, detected=detected)
    line_R.update_line(right_fit_pixel, right_fit_meter, detected=detected)
    
    # AVG the lane-lines data detected over N iterations for both Left and Right lanes.
    line_L.last_fit_pixel = left_fit_pixel = line_L.best_fit_pixels
    line_L.last_fit_meter = left_fit_meter = line_L.best_fit_meters

    line_R.last_fit_pixel = right_fit_pixel = line_R.best_fit_pixels
    line_R.last_fit_meter = right_fit_meter = line_R.best_fit_meters

    # Generate x and y values for plotting
    ploty = np.linspace(0, height - 1, height)
    left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
    right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

    # Create an image to draw on and an image to show the selection window
    img_fit = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255
    window_img = np.zeros_like(img_fit)

    # Color in left and right line pixels
    img_fit[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    img_fit[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(img_fit, 1, window_img, 0.3, 0)

    if verbose:
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.show()

    return line_L, line_R, img_fit


def unwarp_lines(undist, color_warp, line_L, line_R, Minv, keep_history, verbose=False):

    h, w, c = undist.shape

    left_fit = line_L.average_fit if keep_history else line_L.last_fit_pixel
    right_fit = line_R.average_fit if keep_history else line_R.last_fit_pixel

    # Generate x and y values for plotting
    ploty = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + \
        right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undist, dtype=np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_zero, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_zero, Minv, (w, h))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    if verbose:

        plt.imshow(cv2.cvtColor(result, code=cv2.COLOR_BGR2RGB))
        figManager = plt.get_current_fig_manager()  # to control the figure to be showen
        # maximaize the window of the plot to cover the whole screen
        figManager.window.showMaximized()
        plt.show()

    return result


if __name__ == '__main__':

    line_L, line_R = Line(buffer_length=10), Line(buffer_length=10)

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(
        calib_images_directory='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)

        img_undistorted = undistort(img, mtx, dist, verbose=False)

        img_binary, closing, opening = binarize_image(
            img_undistorted, verbose=False)

        img_birdview, M, Minv = birdview(img_binary, verbose=False)

        line_L, line_R, img_lanes = find_lane_pixels(
            img_birdview, line_L, line_R, nwindows=9, verbose=True)

        draw_lines_on_image = unwarp_lines(
            img_undistorted, img_lanes, line_L, line_R, Minv, keep_history=False, verbose=True)
