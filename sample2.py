import cv2,sys
from silatra import Gesture

gesture_video = cv2.VideoCapture('samples/please-give-me-your-pen.avi')
gesture = Gesture(using_stabilization=True)
while True:
    ret, frame = gesture_video.read()
    if not ret: break
    gesture.add_frame(frame)
    cv2.imshow('Gesture',frame)
    k = cv2.waitKey(1)
    if k==ord('q'): break

print('The recognised gesture is -> '+gesture.classify_gesture())
cv2.destroyAllWindows()
gesture_video.release()
