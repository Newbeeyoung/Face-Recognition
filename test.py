import numpy as np
import cv2

img=cv2.imread("jiang1.jpeg",0)
img_m=cv2.flip(img,1)
print(img.shape)
cv2.imshow("img",img)
cv2.imshow("img1",img_m)
cv2.imwrite("jiang3.jpg",img_m)
cv2.waitKey(0)