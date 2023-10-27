import cv2
import numpy as np
import math


def merge_contours(contours, threshold_distance=10):
    merged_contours = []

    for contour in contours:
        if not merged_contours:
            merged_contours.append(contour)
        else:
            merged = False
            for i, merged_contour in enumerate(merged_contours):
                c1 = tuple(contour[0][0])
                c2 = tuple(merged_contour[0][0])
                distance = np.linalg.norm(np.array(c1) - np.array(c2))
                if distance < threshold_distance:
                    # Merge the two contours
                    merged_contours[i] = np.vstack((merged_contour, contour))
                    merged = True
                    break
            if not merged:
                merged_contours.append(contour)

    return merged_contours
class TunaCoordinate:



    def firstTunaCoordinate(self):


        net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb",
                                            "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        img = cv2.imread('tuna.jpg')
        height, width, _ = img.shape
        # Convert the image to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        # edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Adjust thresholds as needed

        # black_image = np.zeros((height, width, 1), np.uint8)

        blob = cv2.dnn.blobFromImage(img, swapRB=True)

        net.setInput(blob)

        boxes, masks = net.forward(["detection_out_final", "detection_masks"])
        detection_count = boxes.shape[2]

        all_contours = []
        for i in range(detection_count):
            box = boxes[0, 0, i]
            # print(box)
            class_id = box[1]
            score = box[2]
            # print(score)
            if score < 0.5:
                continue
            # get box Cordinates
            x = int(box[3] * width)
            y = int(box[4] * height)
            x2 = int(box[5] * width)
            y2 = int(box[6] * height)
            # print(x)
            roi = img[y:y2, x:x2]
            # print(roi)
            roi_height, roi_width, _ = roi.shape


            mask = masks[i, int(class_id)]

            # print(mask)


            mask = cv2.resize(mask, (roi_width, roi_height))




            _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

            # Get mask cordinate
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print('contours')
            ListX = []
            ListY = []
            for contour in contours:



                for point in contour:
                    x_coord, y_coord = point[0]
                    ListX.append(x_coord)
                    ListY.append(y_coord)
                    # intnum = int(x_coord)
                    # maxi = max(intnum)
                    # # print(x_coord)
                    # # print(f"Object {i} - Coordinate: ({x + x_coord}, {y + y_coord})")
                    coordinate_text = f"({x_coord}, {y_coord})"
                    # maximum_value = max(x_coord)
                    minimum_value = min(coordinate_text)
                    # max = max(map(max,coordinate_text))
                    # print(maxi)
                    # cv2.putText(img, coordinate_text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),
                    #             1)
                    # print(point)
                    # Calculate the distance between two points
                    if len(contour) >= 2:

                        point1 = contour[0][0]
                        print(point1)# First point in the contour
                        point2 = contour[1][0]
                        # print(point2)# Second point in the contour
                        x1, y1 = point1[0] + x, point1[1] + y
                        # print(x1,y1)
                        x2, y2 = point2[1] + x, point2[0] + y

                        # distance = calculate_distance(x1, y1, x2, y2)
                        # print(distance)
                        coordinate_text = f"({x_coord}, {y_coord})"
                        # cv2.putText(img, coordinate_text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        #             (255, 255, 0), 1)
                        # cv2.approxPolyDP()
                        all_contours.extend(contours)

                    # Merge the contours to group objects that belong to the same fish
                # merged_contours = merge_contours(all_contours)
                # for contour in merged_contours:
                #     for point in contour:
                #         x_coord, y_coord = point[0]
                #         coordinate_text = f"({x_coord}, {y_coord})"
                #         cv2.putText(img, coordinate_text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                #                         (255, 255, 0), 1)
                # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
                # cv2.imwrite('annotated_tuna.png', img)
                # cv2.imshow('Annotated Tuna', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                        # Display the distance on the image
                        # cv2.putText(img, distance_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        # Display or save the extracted fish




                    cv2.drawContours(img, [contour + (x, y)], -1, (0, 255, 0), 2)
                # maxX =max(ListX)
                # minX =min(ListX)
                # maxY =max(ListY)
                # minY =min(ListY)
                # print(maxX,minX,maxY,minY)



        cv2.imshow("Detected Objects with Polygons and Coordinates", img)
        cv2.waitKey(0)

        print('hele')

TunaCoordinate().firstTunaCoordinate()