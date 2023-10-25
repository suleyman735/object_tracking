import cv2

# from tracker import *



import numpy as np


# from object_detection import ObjectDetection
# od  = ObjectDetection()




cap = cv2.VideoCapture('file/training_video.mp4')
tracker = cv2.Tracker

object_detector = cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)
while True:
    ret,frame = cap.read()
    height,width,_ = frame.shape

    # Extract Regiaon of interest
    roi = frame[340:720,500:800]

    mask = object_detector.apply(roi)
    _,mask = cv2.threshold(mask,254,255,cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detection = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>100:
            # cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),3)
            detection.append([x,y,w,h])
        # print(detection)

        # print(frame)



    cv2.imshow("Frame",frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('roi', roi)
    key =  cv2.waitKey(30)
    if  key == 27:
        break

cap.release()
cv2.destroyAllWindows()


