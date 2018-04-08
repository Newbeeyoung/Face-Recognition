import numpy as np
import cv2
from face import Face

cap=cv2.VideoCapture(0)

face_reco=Face()
face_reco.train()
# face_reco.load_train_data()

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while(1):
    # img=cv2.imread("jiang1.jpeg")
    name = "unknown"
    ret,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, minSize=(20,20))
    ab=0
    max_w=0
    max_h=0
    max_x=0
    max_y=0
    for (x,y,w,h) in faces:
        ac=w*h
        if ac>ab:
            ab=ac
            max_w=w
            max_h=h
            max_x=x
            max_y=y

    extend_w=int(max_w/6.0)
    extend_h=int(max_h/4.0)
    cv2.rectangle(img, (max_x-extend_w,max_y-extend_h),(max_x+max_w+extend_w,max_y+max_h+extend_h),(255,0,0),2)

    head=gray[max_y-extend_h:max_y+max_h+extend_h,max_x-extend_w:max_x+max_w+extend_w]
    # cv2.imwrite("yi3.png",head)

    if head.shape[0]*head.shape[1]>0:
        name=face_reco.test(head)
        cv2.imshow("head", head)
        cv2.putText(img,name,(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    cv2.imshow("img",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
