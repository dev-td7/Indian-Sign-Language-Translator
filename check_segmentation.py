from utils import segment
import cv2, numpy as np

print('Press q/Q to quit')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

lower = np.array([0,140,60],np.uint8)
upper = np.array([255,180,127],np.uint8)

while True:
    _, frame = cap.read()
    mask = segment(frame)
    '''
    Normal segmentation using YCRCB
    mask = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(mask, lower, upper)
    '''
    cv2.imshow('Original', frame)
    cv2.imshow('Segmentation mask', mask)
    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('Q'): break

cv2.destroyAllWindows()
cap.release()
