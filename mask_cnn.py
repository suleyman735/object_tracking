import os
import sys

import cv2
import numpy as np
import math


# Function to calculate the Euclidean distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

coordinate= [[5,5],[3,3],[5,6],[9,8]]
def calculate_area(coords):
    print('jjj')
    n = len(coords)
    area = 0

    for i in range(n):
        x1,y1 = coords[i]
        x2,y2 = coords[(i+1) % n]
        area += (x1*y2 - x2*y1)
        # print(area)
    area = abs(area) / 2.0
    return area

# print(calculate_area(coordinate))

net  = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")




img = cv2.imread('tuna.jpg')
height, width, _ = img.shape

# Black image
black_image = np.zeros((height,width,1),np.uint8)
# detect image

blob = cv2.dnn.blobFromImage(img,swapRB=True)
# print(blob)
net.setInput(blob)

boxes,masks = net.forward(["detection_out_final","detection_masks"])
detection_count = boxes.shape[2]
print(boxes)
# print(masks)
allarray = []
for i in range(detection_count):

    box = boxes[0,0,i]
    # print(box)
    class_id = box[1]

    score = box[2]
    if score<0.5:
        continue

    # get box Cordinates
    x= int (box[3] *width)
    y=int (box[4] *height)
    x2 =int (box[5] * width)
    y2 = int(box[6] * height)


    # roi = img[y:y2,x:x2]
    roi = black_image[y:y2, x:x2]
    # print(roi)
    roi_height,roi_width,_= roi.shape
    # print(roi_height,roi_width)
    # Get the Mask
    mask = masks[i,int(class_id)]
    # print(mask)

    mask = cv2.resize(mask,(roi_width,roi_height))
    _,mask = cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)


    # Get mask cordinate
    contours,_ = cv2.findContours(np.array(mask,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for i in contours:
        for d in i:
            x1,y1=d[0]
            # print(x1 , y1)



    for contour in contours:
        # Extract and display the coordinates of each point in the contour
        for point in contour:
            # print(point[0])
            x_coord, y_coord = point[0]
            # print(x_coord)
            # print(f"Object {i} - Coordinate: ({x + x_coord}, {y + y_coord})")
            coordinate_text = f"({x_coord}, {y_coord})"
            # print(coordinate_text)
            cv2.putText(img, coordinate_text, (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Calculate the distance between two points
            if len(contour) >= 2:
                # print(contour)
                # print(contour)
                # for i in contour:
                #     for d in range(len(i)):
                #      # print(i[d])
                #      g,h = i[d]
                #
                #     # print(g,h)
                # x,y,w,h= cv2.boundingRect()
                point1 = contour[0][0]
                # print(point1)# First point in the contour
                point2 = contour[1][0]
                # print(point2)# Second point in the contour
                x1, y1 = point1[0] + x, point1[1] + y
                # print(x1,y1)
                x2, y2 = point2[1] + x, point2[0] + y

                distance = calculate_distance(x1, y1, x2, y2)
                # print(distance)
                distance_text = f"Distance: {distance:.2f}"
                # Display the distance on the image
                cv2.putText(img, distance_text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            cv2.drawContours(img, [contour + (x, y)], -1, (0, 255, 0), 2)

        # Draw the contour on the original image
        cv2.drawContours(img, [contour + (x, y)], -1, (0, 255, 0), 2)

        # cv2.imshow("Detected Objects", black_image)
        # cv2.waitKey(0)


    for cnt in contours:
        # print(cnt)
        cv2.fillPoly(roi,[cnt],(255,0,0))
        # cv2.line(roi,[cnt])

        flattened_list = [item for sublist in cnt for item in sublist]
        label = f"({x}, {y}) - ({x2}, {y2})"
        # Display the label on the original image
        cv2.putText(black_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        # Show the original image with coordinates
        # cv2.imshow("Detected Objects", black_image)

        cv2.imshow("Detected Objects with Polygons and Coordinates", img)
        cv2.waitKey(0)
        # print(cnt)
        # print(calculate_area(flattened_list))
        # allarray.append(cnt)

        # print(cnt)
    #     print(cnt)
    #     cv2.imshow("roi", roi)
    #     cv2.waitKey(0)
    #
    #
    #
    #
    # print(mask)




# cv2.rectangle(img,(x, y), (x2, y2), (255, 255, 0), 3)
    # cv2.imshow("Mask", mask)
# cv2.imshow("Black image", black_image)
# Create a text label with coordinates

    # cv2.imshow("roi",roi)
# cv2.waitKey(0)
    # print(x,y)




# print(box)
# x = box[3]
# print(x)

# print(boxes)

# cv2.imshow("Image",img)
# cv2.imshow("Black image",black_image)
# cv2.waitKey(0)


