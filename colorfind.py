import cv2


class colorFind():
    def colorCapture(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            _,frame = cap.read()
            hsv_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            # middle dot  - Pick Pixel value
            height,width,_ = frame.shape
            # print(height)
            cx = int(width / 2)
            ch = int(height / 2)
            pixel_center=hsv_frame[ch,cx]
            color_bg_frame = frame[ch, cx]
            b,g,r= int(color_bg_frame[0]),int(color_bg_frame[1]),int(color_bg_frame[2])
            color = 'red'
            strb = str(b)
            strg = str(g)
            strr = str(r)
            wholeColor = strb+'-'+strg+'-'+strr

            print(wholeColor)

            cv2.putText(frame,wholeColor,(10,50),0,1,(b,g,r),2)
            cv2.circle(frame,(cx,ch),5,(b,g,r),10)


            cv2.imshow("Frame",frame)
            key = cv2.waitKey(1)
            if key ==27:
                break

        cap.release()
        cv2.destroyAllWindows()

colorFind().colorCapture()