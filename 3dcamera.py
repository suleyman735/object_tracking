import cv2
import numpy as np
from matplotlib import pyplot as plt
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
colors = np.random.randint(0, 255, (80, 3))
# 2 connection to your webcam
cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)


# ret1,frame1 = cap1.read()
# ret2,frame2 = cap2.read()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 & ret2:
        break

    img1 = cv2.resize(frame1, (650, 550))
    img2 = cv2.resize(frame2, (650, 550))
    height1, width1, _ = img1.shape
    height2, width2, _ = img2.shape
    black_image1 = np.zeros((height1, width1, 3), np.uint8)
    black_image2 = np.zeros((height2, width2, 3), np.uint8)
    black_image1[:] = (150, 150, 0)

    print(black_image1)




while cap1.isOpened() & cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    cv2.imshow('webcam1',frame1)
    cv2.imshow('webcam2', frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()

cv2.destroyAllWindows()
cap1.isOpened()
