import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from threshold_functions import abs_sobel_thresh
from threshold_functions import mag_thresh
from threshold_functions import dir_threshold
from threshold_functions import hls_select
from threshold_functions import hsv_select

# Read in an image
image = mpimg.imread('test_images/test4.jpg')

ksize = 5;

# Run the function
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))

grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))

# Run the function
mag_binary = mag_thresh(image, sobel_kernel=9, mag_thresh=(30, 100))

# Run the function
dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))

# Run the function
hls_s_binary = hls_select(image, thresh=(100, 255), channel='S')

# Run the function
hls_h_binary = hls_select(image, thresh=(70, 255), channel='H')

# Run the function
hsv_v_binary = hsv_select(image, thresh=(230, 255), channel='V')

combined = np.zeros_like(dir_binary)
#combined[((gradx == 1) & (grady == 0)) | ((mag_binary == 1) | (dir_binary == 1)) | (hls_s_binary == 1)] = 1
combined[((gradx == 1) & (grady == 0)) | (hls_s_binary == 1) & (hls_h_binary == 0) | (hsv_v_binary == 1)] = 1

color_binary = np.dstack(( np.zeros_like(gradx), gradx, hls_s_binary)) * 255


# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(color_binary)
ax2.set_title('Stacks.', fontsize=50)
ax3.imshow(combined, cmap='gray')
ax3.set_title('Thresholded Grad. Combined.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.figure()
plt.imshow(hsv_v_binary)

