import cv2
import  numpy as np
# import conveyor_lib

class bigSize():
    def findBigSizeObject(self):
        cap = cv2.VideoCapture(1)
        while True:
            _, frame = cap.read()
            # gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


            belt = frame[400:800,890:1250]
            gray_belt = cv2.cvtColor(belt, cv2.COLOR_BGR2GRAY)
            _,treshold = cv2.threshold(gray_belt,210,255,cv2.THRESH_BINARY)

            # detect the cups
            contours, _ = cv2.findContours(treshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # print(_)

            for cnt in contours:
                # pass
                # print(cnt)

                (x,y,w,h) = cv2.boundingRect(cnt)

                # Calculate area
                area = cv2.contourArea(cnt)


                # Disinguish small and big area in object
                # if area > 3000:
                #
                #
                #     cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # elif 3300 < area < 6000:
                #
                #     cv2.rectangle(belt, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(belt, str(area), (x, y), 1, 1, (0, 255, 0))


                cv2.rectangle(belt,(x,y),(x+w,y+h),(0,255,0),2)
                print(cnt)





            cv2.imshow("Frame",frame)
            # cv2.imshow("whitePaper", belt)
            cv2.imshow('GrayFrame',gray_belt)
            cv2.imshow('treshold', treshold)

            key = cv2.waitKey(1)
            if key ==27:
                break


        cap.release()
        cv2.destroyAllWindows()

bigSize().findBigSizeObject()

