# https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/

import cv2
import numpy as np
import time
import imutils
from imutils.video import FileVideoStream
import pickle


# define a video capture object
fvs = FileVideoStream(2).start()

# color range for object of interest
# this must be tuned for our setting!
targetColorLower = (0, 0, 0)
targetColorUpper = (85, 85, 85)

iRot = np.array(pickle.load(open("iRot.pkl", "rb")))
iCam = np.array(pickle.load(open("iCam.pkl", "rb")))
trans = np.array(pickle.load(open("trans.pkl", "rb")))

def pixel2xyz(image_point, z = -0.033):

    uvPoint = np.ones((3, 1))
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, trans)

    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    xyz = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - trans))

    return xyz


while fvs.more():

    # Capture the video frame
    frame = fvs.read()

    # blur it, and convert it to the HSV color space
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the target color, then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, targetColorLower, targetColorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing centroid
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(frame, center, 10, (0, 128, 255), 3)
        print(center, pixel2xyz(center))


    # print the center of the centroid
    pickle.dump(center, open("center.pkl", "wb" ))

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()
