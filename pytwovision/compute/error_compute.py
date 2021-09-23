import cv2 as cv

def re_projection_error(objpoints, rvecs, tvecs, mtx, imgpoints, dist):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error/len(objpoints)