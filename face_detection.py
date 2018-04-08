import numpy as np
import cv2
from face import Face

cap=cv2.VideoCapture(0)

face_reco=Face()
face_reco.load_train_data()

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(1):
    img=cv2.imread("jiang1.jpeg")

    # ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, minSize=(20,20))

    for (x,y,w,h) in faces:
        extend_w=int(w/6.0)
        extend_h=int(h/4.0)
        cv2.rectangle(img, (x-extend_w,y-extend_h),(x+w+extend_w,y+h+extend_h),(255,0,0),2)
        head=gray[y-extend_h:y+h+extend_h,x-extend_w:x+w+extend_w]

    name=face_reco.test(head)
    cv2.putText(img,name,(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    cv2.imshow("img",img)
    cv2.imshow("head",head)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
