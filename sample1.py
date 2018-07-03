import silatra, cv2, sys, numpy as np

test_image = cv2.imread('samples/gesture_pose_fist.png')
 
try:
    img = test_image.copy()
    del(img)
    result = silatra.recognise_hand_pose(test_image, model_path='Models/silatra_gesture_signs.sav')
    print('The recognised Hand pose is -> '+result)
except AttributeError: print('Image file does not exist. Please check the image path', file=sys.stderr)
