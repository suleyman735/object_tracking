import cv2
import numpy as np
from sahi.models import yolov5

h, w = None, None
net = cv2.dnn.readNet('dnn/yolov4.weights','dnn/yolov4.cfg')
classes = []

with open('coco.names','r') as f:
    classes = f.read().splitlines()

# Getting only output layer names that we need from YOLO
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize colours for representing every detected object
colours = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
# print(colours)

# print(classes)


cap = cv2.VideoCapture('tellofintuna.mp4')
# height, width, _ = cap.read()


while True:

    _,frame = cap.read()
    if not _:
        break
    # Getting dimensions of the frame for once as everytime dimensions will be same]
    img = cv2.resize(frame, (650, 550))
    blob = cv2.dnn.blobFromImage(img, 1.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # print(blob)
    net.setInput(blob)
    # Perform forward pass
    detections = net.forward()

    for detection in detections:
        confidence = detection[2]
        print(detection)
        print(confidence)


    height, width, _ = img.shape





    # boxes, masks = net.forward(["detection_out_final", "detection_masks"])


    # Detection objetc on frame
    # od.detect(frame)


    cv2.imshow('Frame',frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

