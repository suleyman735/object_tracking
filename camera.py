import cv2
from matplotlib import pyplot as plt

# 2 connection to your webcam
cap = cv2.VideoCapture(0)
ret,frame = cap.read()
# print(ret)
# cv2.imshow("Frame", frame)
# plt.imshow(frame)

# cap.release()
print(frame)

def take_phote():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    cv2.imwrite('first.jpg',frame)
    cap.release()
# take_phote()


# rendering in real time

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('webcam',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
cap.isOpened()