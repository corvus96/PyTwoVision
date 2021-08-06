import requests
import cv2
import numpy as np

def frameByFrame(url):
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    return cv2.imdecode(img_arr, -1)
        

if __name__ == "__main__":
    while True:
        img = frameByFrame("http://192.168.0.111:8080/shot.jpg")
        cv2.imshow("AndroidCam", img)
        if cv2.waitKey(1) == 27:
            break