import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt



# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, orient='xy', sobel_kernel=3, thresh=(0, 255), verbose=False):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 5) Blur the image
    # 5) apply Otsu's thresholding after Gaussian filtering
    # 6) Return this mask as your binary_output image

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel  = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel  = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    if orient == 'xy':
        abs_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        abs_sobel = np.sqrt(abs_sobel_x ** 2 + abs_sobel_y ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel/np.max(abs_sobel))

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(scaled_sobel,(5,5),0)
    _,binary_output = cv2.threshold(blur,thresh[0],thresh[1],cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #binary_output = np.zeros_like(scaled_sobel)
    #binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    if verbose:
        plt.imshow(binary_output, cmap='gray')
        plt.show()
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #find the gradiant in x
    sobelx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    #find the gradiant in y
    sobely  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #calculate the gradient magnitude in both directions (x&y)
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
    #scale to 255 and convert to uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    
	#creat a binary mask where thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #gradiant in x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    #gradient in y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    #absuloute value of sobelx
    abs_sobelx = np.absolute(sobelx)
    #absuloute value of sobely
    abs_sobely = np.absolute(sobely)
    #magnitude of gradient in both directions
    sobelxy = np.sqrt(sobelx**2+sobely**2)
    #gradient direction
    grad_dir = np.arctan2(abs_sobely, abs_sobelx)
    
    #creat a binary mask where thresholds are met
    binary_output = np.zeros_like(grad_dir)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255),channel='S'):
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    
    #convert image from RGB to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    #get the H channel from the HLS image
    H = hls[:,:,0]
    #get the L channel from the HLS image
    L = hls[:,:,1]
    #get the S channel from the HLS image
    S = hls[:,:,2]
    
    #apply threshold to the S channel and produce a binary image
    binary_output = np.zeros_like(H)
    if channel == 'H':
        binary_output[(H > thresh[0]) & (H <= thresh[1])] = 1
    if channel == 'L':
        binary_output[(L > thresh[0]) & (L <= thresh[1])] = 1
    if channel == 'S':
        binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output

def hsv_select(img, thresh=([0, 70, 70], [50, 255, 255]),channel='S', verbose=False):
    # 1) Convert to HSV color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    
    # define range of blue color in HSV
    lower_b = np.array(thresh[0])
    upper_b = np.array(thresh[1])

    #convert image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    #get the H channel from the HLS image
    H = hsv[:,:,0]
    #get the S channel from the HLS image
    S = hsv[:,:,1]
    #get the V channel from the HLS image
    V = hsv[:,:,2]
    
    #apply threshold to the S channel and produce a binary image
    binary_output = np.zeros_like(H)
    if channel == 'H':
        binary_output[(H > thresh[0][0]) & (H <= thresh[1][0])] = 1
    if channel == 'S':
        binary_output[(S > thresh[0][1]) & (S <= thresh[1][1])] = 1
    if channel == 'V':
        binary_output[(V > thresh[0][2]) & (V <= thresh[1][2])] = 1
    if channel == 'all':
        # Threshold the HSV image to get only range threshold colors
        binary_output = cv2.inRange(hsv, lower_b, upper_b )
        '''binary_output[(H > thresh[0][0]) & (H <= thresh[1][0]) & 
                      (S > thresh[0][1]) & (S <= thresh[1][1]) & 
                      (V > thresh[0][2]) & (V <= thresh[1][2]) ] = 1'''

    if verbose:
        plt.imshow(binary_output, cmap='gray')
        plt.show()

    return binary_output

def histo_image(image, verbose=False):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    histo_global = cv2.equalizeHist(gray)

    _, histo = cv2.threshold(histo_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    if verbose:
        plt.imshow(histo, cmap='gray')
        plt.show()

    return histo

def binarize_image(img, verbose=False):
    """
    Convert an input frame to a binary image that highlights lanes as much as possible

    :param img: input color frame
    :param show: if True, show resulting images
    :return: binarized image
    """
    h, w = img.shape[:2]

    #creat an empty image with the same size as the passed frame to the function
    binary_output = np.zeros(shape=(h, w), dtype=np.uint8)

    #using HSV, Find yellow lanes in the image (min [0, 70, 70] and max [50, 255, 255] were selected to detect yellow at all conditions in the image)
    HSV_yellow_lanes = hsv_select(img, thresh=([0, 70, 70], [50, 255, 255]), channel='all', verbose=False)

    #add the yellow mask to the binary image
    binary_output = np.logical_or(binary_output, HSV_yellow_lanes)

    #using Histogram Equalization, Find white lanes in the image 
    histo_white_lanes = histo_image(img, verbose=False)

    #add the white mask to the binary image
    binary_output = np.logical_or(binary_output, histo_white_lanes)

    #apply sobel mask to the image
    sobel_mask = abs_sobel_thresh(img, orient='xy', sobel_kernel=9, thresh=(0, 255), verbose=False)

    #add the sobel mask to the binary image
    binary_output = np.logical_or(binary_output, sobel_mask)

    #using HLS, Find lanes in the image
    hls_s_binary = hls_select(img, thresh=(150, 255), channel='S')

    #add the sobel mask to the binary image
    binary_output = np.logical_or(binary_output, hls_s_binary)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_output.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if verbose:
        plt.figure()
        plt.imshow(binary_output, cmap='gray')
        plt.show()

        plt.imshow(closing, cmap='gray')
        plt.show()

        cv2.waitKey(1500)

    return binary_output, closing


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary

if __name__ == '__main__':

    test_images = glob.glob('test_images/*.jpg')
    for test_image in test_images:
        img = cv2.imread(test_image)
        binary_output, closing = binarize_image(img=img, verbose=False)
        #plt.figure()
        plt.imshow(binary_output, cmap='gray')
        plt.show()