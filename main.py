import collections
import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from calibration_functions import calibrateCamera_SLOW, undistort
from globals import xm_per_pix, ym_per_pix, N
from perspective_function import birdview
from threshold_functions import binarize_image
from lines_functions import find_lane_pixels, unwarp_lines, Line



# counter of frames processed (when processing video files)
processed_frames = 0

line_L = Line(buffer_length=N)  # line on the left of the lane
line_R = Line(buffer_length=N)  # line on the right of the lane


def find_offset(line_L, line_R, img):
    """
    Assuming the camera is mounted at the center of the car, such that
    the lane center is the midpoint at the bottom of the image
    between the two lines you've detected. 
    The offset of the lane center from the center of the image 
    (converted from pixels to meters) is your distance from the center of the lane.

    :param line_L: detected left lane data
    :param line_R: detected right lane data
    :param img_width: width of the undistorted img coming from the camera
    :return: offset of the lane center from the center of the image
    """

    h, w, c = img.shape

    if line_L.detected and line_R.detected:
        line_lt_bottom = line_L.last_fit_pixel[0] * h ** 2 + \
            line_L.last_fit_pixel[1] * h + line_L.last_fit_pixel[2]
        line_rt_bottom = line_R.last_fit_pixel[0] * h ** 2 + \
            line_R.last_fit_pixel[1] * h + line_R.last_fit_pixel[2]
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = w / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    print(offset_meter)
    return offset_meter


def pipeline(img):
    """
    This is a pipeline function where the image go through different filters to find the lanes in it
    :param img: input coloured image
    :return: output image with highligthed area between the detected lanes
    """

    global line_L, line_R, processed_frames

    img_undistorted = undistort(img, mtx, dist, verbose=False)

    img_binary, closing, opening = binarize_image(
        img_undistorted, verbose=False)

    img_birdview, M, Minv = birdview(img_binary, verbose=False)

    line_L, line_R, img_lanes = find_lane_pixels(
        img_birdview, line_L, line_R, nwindows=9, verbose=False)

    img_output = draw_lines_on_image = unwarp_lines(
        img_undistorted, img_lanes, line_L, line_R, Minv, keep_state=False, verbose=False)

    offset_meter = find_offset(line_L, line_R, img_output)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(img_output, img_binary, img_birdview, img_lanes, line_L, line_R, offset_meter)


    blend_output = prepare_out_blend_frame(img_output, img_binary, img_birdview, img_lanes, line_L, line_R, offset_meter)
    return blend_output

def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images

    :param blend_on_road: color image of lane blend onto the road
    :param img_binary: thresholded binary image
    :param img_birdeye: bird's eye view of the thresholded binary image
    :param img_fit: bird's eye view with detected lane-lines highlighted
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param offset_meter: offset from the center of the lane
    :return: pretty blend with all images and stuff stitched
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(thumb_w+2*off_x, int(h*0.69)), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[2*off_y+thumb_h:2*(thumb_h+off_y), off_x:off_x+thumb_w, :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[3*off_y+2*thumb_h:3*(thumb_h+off_y), off_x:off_x+thumb_w, :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_L.radius_of_curvature_meters, line_R.radius_of_curvature_meters])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (off_x, 3*(thumb_h+off_y)+2*off_y), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (off_x, 3*(thumb_h+off_y)+4*off_y), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


    return blend_on_road



if __name__ == '__main__':

    line_L, line_R = Line(buffer_length=10), Line(buffer_length=10)

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(
        calib_images_directory='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)
        blend = pipeline(img)

        #cv2.imwrite('output_images/{}'.format(test_img), blend)
        print('Radius of Curvature = ',np.mean([line_L.radius_of_curvature_meters,line_R.radius_of_curvature_meters]), 'm')
        plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        plt.show()