import cv2 as cv

def re_projection_error(objpoints, rvecs, tvecs, mtx, imgpoints, dist):
    """Calculates the error when calibrating a camera. 
    when the error approaches zero, 
    the more accurate are the parameters found. 
    Given the intrinsic, distortion, rotation and translation matrices, we must first transform the 
    translation matrices, we must first transform the object point to the point of the camera. 
    the object point to image point using cv.projectPoints().
    Then, we will calculate the absolute norm between what we obtained with our transformation and the algorithm. 
    obtained with our transformation and the corner search algorithm. 
    To find the average error, we calculate the arithmetic average
    of the errors calculated for all calibration images.
    
    Args:
        objpoints: array of object points expressed in the world coordinate frame. A 1-channel 3xN/Nx3 or 3-channel 1xN/Nx1, where N is the number of points in the view.
        rvecs: the rotation vector (Rodrigues) which, together with tvec, performs a change of base from the world coordinate system to the camera, see calibrateCamera for more details.
        tvecs: The translation vector, see the description of the parameters above.
        mtx: The intrinsic matrix of the camera.
        imgpoints: reference points to compare with the reprojected objpoints.
        dist: Input vector of the distortion coefficients.

    Returns
        a float with the mean error with L2 norm.
    """
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    return mean_error/len(objpoints)