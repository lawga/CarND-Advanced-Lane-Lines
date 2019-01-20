import collections
import glob
import os
import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

from calibration_functions import calibrateCamera_SLOW, undistort
from globals import N, xm_per_pix, ym_per_pix
from lines_functions import Line, find_lane_pixels, unwarp_lines, get_fits_by_previous_fits
from perspective_function import birdview
from threshold_functions import binarize_image

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
        offset_pixels = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pixels
    else:
        offset_meter = -1

    # print(offset_meter)
    return offset_meter, offset_pixels


def blend_output_image(draw_on_road, img_binary, img_birdview, img_fit, offset_meter, offset_pixels):
    """
    Prepare the final output blend, showing the main pipeline images on the lesft side of the screen and also showing Curveture and offset on the lane region

    :param draw_on_road: color image of lane drawn onto the road
    :param img_binary: thresholded binary image
    :param img_birdview: bird's view of the thresholded binary image
    :param img_fit: bird's view with detected lane-lines highlighted
    :param line_L: detected left lane-line with all its properties (2nd order polymomial curve, binary raw data...etc)
    :param line_R: detected right lane-line with all its properties (2nd order polymomial curve, binary raw data...etc)
    :param offset_meter: offset is the distance between the center of the line and the center of the image
    :return: an image of the road with some of the outpouts of the pipleline stitched to one corner
    """
    hight, width = draw_on_road.shape[:2]

    # The size of the stiched images with respect to the big image
    thumb_ratio = 0.2 
    
    # calculate the hight and width of the images that will be stiched
    thumb_hight, thumb_width = int(thumb_ratio * hight), int(thumb_ratio * width) 

    off_x, off_y = 20, 15

    # add a gray rectangle as a frame to highlight the left area of the screen
    mask = draw_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(
        thumb_width+2*off_x, int(hight*0.75)), color=(0, 0, 0), thickness=cv2.FILLED)
    draw_on_road = cv2.addWeighted(
        src1=mask, alpha=0.2, src2=draw_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_width, thumb_hight))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    draw_on_road[off_y:thumb_hight+off_y, off_x:off_x+thumb_width, :] = thumb_binary

    # add thumbnail of bird's view
    thumb_birdview = cv2.resize(img_birdview, dsize=(thumb_width, thumb_hight))
    thumb_birdview = np.dstack(
        [thumb_birdview, thumb_birdview, thumb_birdview]) * 255
    draw_on_road[2*off_y+thumb_hight:2 *
                  (thumb_hight+off_y), off_x:off_x+thumb_width, :] = thumb_birdview

    # add thumbnail of bird's view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_width, thumb_hight))
    draw_on_road[3*off_y+2*thumb_hight:3 *
                  (thumb_hight+off_y), off_x:off_x+thumb_width, :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean(
        [line_L.radius_of_curvature_meters, line_R.radius_of_curvature_meters])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(draw_on_road, 'L Curvature: {:.02f}m'.format(line_L.radius_of_curvature_meters), (off_x, 3*(thumb_hight+off_y)+int(1.5*off_y)), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(draw_on_road, 'R Curvature: {:.02f}m'.format(line_R.radius_of_curvature_meters), (off_x, 3*(thumb_hight+off_y)+3*off_y), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(draw_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (int((width-400)/2), 600), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    # cv2.putText(draw_on_road, 'Offset from center: {:.02f}m'.format(
    #     offset_meter), (off_x, 3*(thumb_hight+off_y)+5*off_y), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(draw_on_road, 'Offset from center: {:.02f}m'.format(
        offset_meter), (int((width-400)/2+offset_pixels), 650), font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    return draw_on_road


def pipeline(img, keep_state=True):
    """
    This is a pipeline function where the image go through different filters to find the lanes in it
    :param img: input coloured image
    :return: output image with highligthed area between the detected lanes
    """

    global line_L, line_R, processed_frames #Declare

    # undistort the image using the coefficients calculated when the camera was calibrated
    img_undistorted = undistort(img, mtx, dist, verbose=False)

    # pass the image to the binarization pipline where yellow and whaite lanes are highlighted
    img_binary, closing, opening = binarize_image(
        img_undistorted, verbose=False)

    # Transform the transformed image to a bird view prespective
    img_birdview, M, Minv = birdview(img_binary, verbose=False)

    
    if processed_frames > 0 and keep_state and line_L.detected and line_R.detected:
        print('L:',line_L.detected,'R:', line_R.detected)
        # use sliding window to find lanes in images and fit the founded points into a 2-degree polynomial curve
        line_L, line_R, img_lanes = get_fits_by_previous_fits(img_birdview, line_L, line_R, verbose=False)
    else:    
        # use preivious sliding window data from old frames to estimate and smooth lanes in images and fit the found lanes
        line_L, line_R, img_lanes = find_lane_pixels(
            img_birdview, line_L, line_R, nwindows=9, verbose=False)

    # draw the lane region (in green) on the original image
    img_output = draw_lines_on_image = unwarp_lines(
        img_undistorted, img_lanes, line_L, line_R, Minv, keep_state=False, verbose=False)

    # compute the offset between the center of the car and the center of the lane
    offset_meter, offset_pixesls = find_offset(line_L, line_R, img_output)

    # stitch on the top of final output images from different steps of the pipeline
    images_in_image = blend_output_image(
        img_output, img_binary, img_birdview, img_lanes, offset_meter, offset_pixesls)

    processed_frames += 1

    return images_in_image


""" if __name__ == '__main__':

    # line_L, line_R = Line(buffer_length=10), Line(buffer_length=10)

    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(
        calib_images_directory='camera_cal')

    # show result on test images
    for test_img in glob.glob('test_images/*.jpg'):

        img = cv2.imread(test_img)
        blend = pipeline(img)

        #cv2.imwrite('output_images/{}'.format(test_img), blend)
        # print('Radius of Curvature = ',np.mean([line_L.radius_of_curvature_meters,line_R.radius_of_curvature_meters]), 'm')
        plt.imshow(cv2.cvtColor(blend, code=cv2.COLOR_BGR2RGB))
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
 """

if __name__ == '__main__':

    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = calibrateCamera_SLOW(
        calib_images_dir='camera_cal')

    # Choose between video mode or image mode
    mode = 'video'  # 'image' or 'video'

    # show result on test videos
    if mode == 'video':

        test_vid_dir = 'test_videos'  # locate the directory of the targeted video
        selector = 'project_video'  # inatial part of the name of the viddeo file, Can be 'project_video' or 'challenge_video' or 'harder_challenge_video' or any other inital of files you manually add to the project
        clip = VideoFileClip('{}.mp4'.format(os.path.join(test_vid_dir, selector))).fl_image(
            pipeline)  # pick the file and pass it to the pipeline
        # save the file and add the number of frames in the video used to smooth the output of each frame
        clip.write_videofile(
            'output_videos/out_{}_{}_{}.mp4'.format(selector, N, time.strftime("%Y%m%d-%H%M%S")), audio=False)

    else:
        # show result on test images
        test_img_dir = 'test_images'  # locate the directory of the targeted images
        # loop throug specific type/name of files in the selected directory
        for test_img in os.listdir(test_img_dir):

            # read image as a BGR image
            img = cv2.imread(os.path.join(test_img_dir, test_img))

            processed_image = pipeline(img)  # pass the image to the pipline

            cv2.imwrite('output_images/{}'.format(test_img), processed_image)

            # print('Radius of Curvature = ',np.mean([line_L.radius_of_curvature_meters,line_R.radius_of_curvature_meters]), 'm')
            # Plot image after converting it from BGR to RGB
            plt.imshow(cv2.cvtColor(processed_image, code=cv2.COLOR_BGR2RGB))
            figManager = plt.get_current_fig_manager()  # to control the figure to be showen
            # maximaize the window of the plot to cover the whole screen
            figManager.window.showMaximized()
            plt.show()
