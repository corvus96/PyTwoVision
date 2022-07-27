import cv2 as cv

def re_projection_error(objpoints, rvecs, tvecs, mtx, imgpoints, dist):
    """Calculate the error when calibrate a camera. 
    when the error is near zero, 
    the more accurate the parameters we found are. 
    Given the intrinsic, distortion, rotation and 
    translation matrices, we must first transform 
    the object point to image point using cv.projectPoints().
    Then, we're gonna calculate the absolute norm between what 
    we got with our transformation and the corner finding algorithm. 
    To find the average error, we calculate the arithmetical mean
    of the errors calculated for all the calibration images.
    Arguments:
        objpoints: Array of object points expressed wrt. 
        the world coordinate frame. 
        A 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, 
        where N is the number of points in the view.
        rvecs: The rotation vector (Rodrigues) that, together with tvec, performs a 
        change of basis from world to camera coordinate system, see calibrateCamera for details.
        tvecs: The translation vector, see parameter description above.
        mtx: Camera intrinsic matrix.
        imgpoints: reference points to compare with objpoints reprojected.
        dist: Input vector of distortion coefficients.
    Returns:
        a float with average error with L2 norm.
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error/len(objpoints)