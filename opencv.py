import cv2
import os
import numpy as np


def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def grayscale_17_levels (image):
    high = 255
    while(1):
        low = high - 15
        col_to_be_changed_low = np.array([low])
        col_to_be_changed_high = np.array([high])
        curr_mask = cv2.inRange(gray, col_to_be_changed_low,col_to_be_changed_high)
        gray[curr_mask > 0] = (high)
        high -= 15
        if(low == 0 ):
            break
image = cv2.imread(os.path.abspath('image.jpg'))
viewImage(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grayscale_17_levels(gray)
viewImage(gray)


def get_area_of_each_gray_level(im):
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    output = []
    high = 255
    first = True
    while (1):


        low = high - 15
        if (first == False):
            to_be_black_again_low = np.array([high])
            to_be_black_again_high = np.array([255])
            curr_mask = cv2.inRange(image, to_be_black_again_low,
                                    to_be_black_again_high)
            image[curr_mask > 0] = (0)

        ret, threshold = cv2.threshold(image, low, 255, 0)
        contours, hirerchy = cv2.findContours(threshold,
                                              cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if (len(contours) > 0):
            output.append([cv2.contourArea(contours[0])])
        cv2.drawContours(im, contours, -1, (0, 0, 255), 3)
        high -= 15
        first = False
        if (low == 0):
            break
    return output


image = cv2.imread(os.path.abspath('image.jpg'))
print(get_area_of_each_gray_level(image))
viewImage(image)