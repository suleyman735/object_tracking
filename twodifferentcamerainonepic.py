import os
import sys

import cv2
import numpy as np

# from Mask_RCNN.mrcnn.visualize import ROOT_DIR

# sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))
from mrcnn import utils
# from keras import engine as KE

from mrcnn import parallel_model as modellib
from mrcnn import visualize
from mrcnn.config import Config
import sys
print(sys.path)
# 2 connection to your webcam


cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)
class tookPicture:
    net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                        "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

    @staticmethod


    def twoCamera(cap1,cap2):




        # while True:
        #     ret1, frame1 = cap1.read()
        #     ret2, frame2 = cap2.read()
        #
        #
        #     print(frame2)
        #     break
        #     if not ret1 & ret2:
        #        break
        while cap1.isOpened() & cap2.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if frame1 is not None and frame2 is not None:
                cv2.imshow('webcam1', frame1)
                cv2.imshow('webcam2', frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            while (True):
                # Capture frame-by-frame
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()

                # Display the resulting frame
                cv2.imshow('frame1', frame1)
                cv2.imshow('frame2', frame2)

                # Press 's' to save the frame
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    # cv2.imwrite('captured_frame.jpg', frame1)
                    # cv2.imwrite('captured_frame2.jpg', frame2)
                    print("Frame captured and saved as captured_frame.png")
                    break


        # Release the capture
        cap1.release()
        cap2.release()

        # cv2.destroyAllWindows()

        # cap1.release()
        # cap2.release()

        cv2.destroyAllWindows()
        cap1.isOpened()

    # @staticmethod
    # def take_phote(cap1,cap2):
    #     # cap = cv2.VideoCapture(0)  # 0 for the default camera, or provide the path to a video file
    #
    #     while (True):
    #         # Capture frame-by-frame
    #         ret1, frame1 = cap1.read()
    #         ret2, frame2 = cap2.read()
    #
    #         # Display the resulting frame
    #         cv2.imshow('frame1', frame1)
    #         cv2.imshow('frame2', frame2)
    #
    #         # Press 's' to save the frame
    #         if cv2.waitKey(1) & 0xFF == ord('s'):
    #             cv2.imwrite('captured_frame1.jpg', frame1)
    #             cv2.imwrite('captured_frame2.jpg', frame2)
    #             print("Frame captured and saved as captured_frame.png")
    #             break
    #
    #     # Release the capture
    #     cap1.release()
    #     cap2.release()
    #     cv2.destroyAllWindows()



        # cap = cv2.VideoCapture(0)
        # ret,frame = cap.read()
        # cv2.imwrite(name,frame)
        # return cap.release()


# tookPicture.twoCamera(cap1,cap2)

class mergePicCordinate:


    def picture1(self):
        net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                            "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        # Load an image from file
        image = cv2.imread('captured_frame.jpg')
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        print(layer_names)
        outputs = net.forward(layer_names)
        for output in outputs:
            for detection in output:
                print(detection[5:])
                scores = detection[5:]

                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust the confidence threshold as needed
                    center_x, center_y, width, height = list(map(int, detection[0:4] * image.shape[1:3]))
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    print(x)

                    # 'x' and 'y' are the top-left coordinates of the bounding box
                    # 'width' and 'height' are the width and height of the bounding box



mergePicCordinate().picture1()

