# things we need:
# ball to play with -- check, got "hockey puck"
# surface of constant color that clashes with ball --- check, cleaned table
# identify the color of the ball --- check, found the hsv for red online
# fixed camera mount --- check, using stand
# collected pairs of xyz and pixel points
# conversion between pixels and x-y --- backdrive robot with puck, record xyz and pixel, then calibrate?

# while True:
#     get puck position from camera (pixel value) -- we have this setup!
#     map to robot coordinates on board (xy value) -- calibration with fixed camera pose
#     end-effector control robot:
#     many different heuristics! here is one
#     align x with x of puck
#     if puck closer than y_min:
#         move forward


import numpy as np
from numpy.linalg import inv
import pickle
import cv2

# collect datapoints
data = pickle.load(open("points.pkl", "rb"))
data = np.asarray(data, dtype=np.float)
xyz_points = np.array(data[:,0:3])
pixel_points = np.array(data[:,3:5])

# camera calibration
size = [640, 480]
focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array([[focal_length, 0, center[0]],[0, focal_length, center[1]],[0, 0, 1]], dtype = "double")
dist_coeffs = np.zeros((4,1), dtype = "double")

# solve for transformation
(success, rotation_vector, translation_vector) = cv2.solvePnP(xyz_points, pixel_points, camera_matrix, dist_coeffs)

# save elements needed to invert
rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
iRot = inv(rotation_matrix)
iCam = inv(camera_matrix)
trans = translation_vector
pickle.dump(iRot, open("iRot.pkl", "wb" ))
pickle.dump(iCam, open("iCam.pkl", "wb" ))
pickle.dump(trans, open("trans.pkl", "wb" ))



# def groundProjectPoint(image_point, z = -0.033):
#
#     uvPoint = np.ones((3, 1))
#     uvPoint[0, 0] = image_point[0]
#     uvPoint[1, 0] = image_point[1]
#
#     tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
#     tempMat2 = np.matmul(iRot, trans)
#
#     s = (z + tempMat2[2, 0]) / tempMat[2, 0]
#     wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - trans))
#
#     return wcPoint
#
# pixel = (385, 211)
# print("Pixel: %s" % (pixel, ))
# print(groundProjectPoint(pixel))
