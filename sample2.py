import cv2,sys
from silatra import Gesture
import time

gesture_video = cv2.VideoCapture('samples/please-give-me-your-pen.avi')
gesture = Gesture(using_stabilization=True)

total_time, num_frames = 0, 0
while True:
    start = time.time()
    ret, frame = gesture_video.read()
    if not ret: break
    gesture.add_frame(frame)
    # cv2.imshow('Gesture',frame)
    end = time.time()
    total_time += (end - start)
    num_frames += 1

    print('Time taken for frame %d = %.3f ms'%(num_frames,(end-start)*1000), end='\r')
    k = cv2.waitKey(1)
    if k==ord('q'): break

print('The recognised gesture is -> '+gesture.classify_gesture())
print('Time taken for recognition = %.3f s'%(total_time))
cv2.destroyAllWindows()
gesture_video.release()
