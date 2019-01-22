import cv2
import numpy as np

def saveTo(what, where, name='newIMAGE.jpg'):
    # what = np.dstack((what, what, what))
    cv2.imwrite('{}/{}'.format(where, name), what)