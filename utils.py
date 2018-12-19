"""
This file contains several Modules of silatra.
"""

def segment(src_img):
    """
    ### Segment skin areas from hand using a YCrCb mask.

    This function returns a mask with white areas signifying skin and black areas otherwise.

    Returns: mask
    """

    import cv2
    from numpy import array, uint8

    blurred_img = cv2.GaussianBlur(src_img,(5,5),0)
    blurred_img = cv2.medianBlur(blurred_img,5)
    
    ycrcb_image = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2YCR_CB)
    lower = array([0,140,60], uint8)
    upper = array([255,180,127], uint8)
    mask = cv2.inRange(ycrcb_image, lower, upper)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def detect_face(image):
    """
    ### Detects face in an image.

    This function takes input a colour image. It creates a rectangle around the face region.
    
    Returns: (1) a tuple (x,y,w,h) where
                    x: X co-ordinate of top left corner of rectangle
                    y: Y co-ordinate of top left corner of rectangle
                    w: Width of rectangle
                    h: Height of rectangle
             (2) True if face was present in the image and False otherwise.
    """

    import dlib, cv2
    from imutils import face_utils

    detector = dlib.get_frontal_face_detector()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    maxArea1 = 0
    faceRect = -1
    foundFace = False

    for (i, rect) in enumerate(rects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if w*h > maxArea1:
            maxArea1 = w*h
            faceRect = (x,y,w,h)
            foundFace = True

    return (faceRect, foundFace)


def eliminate_face(face, foundFace, mask):
    """
    ### Eliminates face and returns a binary mask without face region but containing other skin regions.

    Inputs:
    (1) A tuple (x,y,w,h) signifying a rectangle around a face.
    (2) If face was found, set this to True, else set False.
    (3) Binary mask obtained after performing skin segmentation.

    Returns: Binary mask containing skin areas.
    """

    import numpy as np, cv2
    
    MIN_AREA_THRESHOLD = 300

    HEIGHT, WIDTH = mask.shape

    if foundFace:
        (x,y,w,h) = face
        faceNeckExtraRect = ((int(x+(w/2)-8), int(y+h/2)), (int(x+(w/2)+8), int(y+h+h/4)))
        cv2.rectangle(mask, faceNeckExtraRect[0], faceNeckExtraRect[1], (255,255,255), -1)
        
        tempImg1 = np.zeros((HEIGHT,WIDTH,1), np.uint8)
        cv2.rectangle(tempImg1, (x, y), (x + w, y + h), (0,0,0), -1)
        cv2.rectangle(tempImg1, faceNeckExtraRect[0], faceNeckExtraRect[1], (255,255,255), -1)
    
    _,contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    max_area_of_intersection = -1
    intersectingContour = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area < MIN_AREA_THRESHOLD:
                cv2.drawContours(mask, contours, i, (0,0,0), -1)
                continue
            if foundFace:                
                tempImg2 = np.zeros((HEIGHT,WIDTH,1), np.uint8)
                cv2.rectangle(tempImg1, (x, y), (x + w, y + h), (255,255,255), -1)
                cv2.drawContours(tempImg2, contours, i, (255,255,255), -1)
                tempImg3 = cv2.bitwise_and(tempImg1,tempImg2)
                area_of_intersection = np.sum(tempImg3 == 255)
                if area_of_intersection > max_area_of_intersection:
                    max_area_of_intersection = area_of_intersection
                    intersectingContour = i
        if intersectingContour != -1:
            cv2.drawContours(mask, contours, intersectingContour, (0,0,0), -1)
    return mask


# --- These global variables are required for Object stabilisation ---
import cv2

faceStabilizerMode = "ON"  # This is used to enable/disable the stabilizer using KCF Tracker
trackingStarted = False     # This is used to indicate whether tracking has started or not
noOfFramesNotTracked = 0    # This indicates the no of frames that has not been tracked
maxNoOfFramesNotTracked = 15 # This is the max no of frames that if not tracked, will restart the tracker algo
minNoOfFramesBeforeStabilizationStart = 0
trackerInitFace = (0,0,0,0)
try: tracker = cv2.TrackerKCF_create()
except AttributeError: tracker = cv2.Tracker_create('KCF')

# --- End of declaration ---

def stabilize(foundFace,noOfFramesCollected,img_np,faceRect,mask1):
    '''
    ### Object stabilisation
    
    Helps stabilize the movement in a continuous feed.

    Inputs:
    (1) (Boolean) If face was found in the image to be stabilized
    (2) (Integer) Number of frames collected so far
    (3) Source Image
    (4) (Tuple) (x,y,w,h) signifying a rectangle around the face
    (5) Binary mask obtained after face elimination.

    * Here is the stabilization logic
    *
    * We are stabilizing the person by using face as the ROI for tracker. Thus, in situations where
    * the person moves while the camera records the frames, or if the camera operator's hand shakes, 
    * these false movements wont be detected.
    * We are using `noOfFramesCollected` so as to improve the stabilization results by delaying the
    * tracker initialization

    '''

    import numpy as np
    import cv2
    import imutils

    global faceStabilizerMode, trackingStarted, noOfFramesNotTracked, maxNoOfFramesNotTracked, minNoOfFramesBeforeStabilizationStart, trackerInitFace, tracker

    if not(trackingStarted) and foundFace and noOfFramesCollected >= minNoOfFramesBeforeStabilizationStart:
        trackingStarted = True
        ok = tracker.init(img_np, faceRect)
        trackerInitFace = faceRect
    elif trackingStarted:
        ok, bbox = tracker.update(img_np)
        if ok:
            cv2.rectangle(img_np, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (255,0,0), 2)
            
            rows,cols,_ = img_np.shape
            tx = int(trackerInitFace[0] - bbox[0])
            ty = int(trackerInitFace[1] - bbox[1])
            shiftMatrix = np.float32([[1,0,tx],[0,1,ty]])
            
            img_np = cv2.warpAffine(img_np,shiftMatrix,(cols,rows))
            mask1 = cv2.warpAffine(mask1,shiftMatrix,(cols,rows))

            noOfFramesNotTracked = 0
        else:
            noOfFramesNotTracked += 1
            if noOfFramesNotTracked > maxNoOfFramesNotTracked:
                trackingStarted = False
                noOfFramesNotTracked = 0
        return mask1


def get_my_hand(img_gray, return_contour=False):
    """
    ### Hand extractor

    __DO NOT INCLUDE YOUR FACE IN THE `img_gray`__
    
    This function does the hardwork of finding your hand area in the image.

    Inputs: (1) An Image where skin areas are represented by white and black otherwise.
            (2) return_contour: If True, returns contour of the hand.


    Returns: (1) (Image) Your hand region
             (2) if return_contour parameter is True, hand contour.
    """

    import cv2

    _,contours,_ = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    contour_found = True
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i
        x,y,w,h = cv2.boundingRect(contours[ci])
        hand = img_gray[y:y+h,x:x+w]
    else: contour_found = False
    
    # To display hand image, uncomment the below lines.
    '''
    hand = np.zeros((img_gray.shape[1], img_gray.shape[0], 1), np.uint8)
    cv2.drawContours(hand, contours, ci, 255, cv2.FILLED)
    _,hand = cv2.threshold(hand[y:y+h,x:x+w], 127,255,0)
    '''
    
    if return_contour and contour_found: return (hand, contours[ci])
    elif return_contour: return (None, None)
    elif contour_found: return hand
    else: return False


def extract_features(src_hand, grid=(10,10)):
    """
    ### Uses M x N Grid based fragmentation to extract features from an image.

    Inputs: (1) Image of hand region (2) Tuple (M, N) signifying grid size.
    Returns: List of features extracted from the image.
    """

    import cv2
    from math import ceil

    HEIGHT, WIDTH = src_hand.shape

    data = [ [0 for haha in range(grid[0])] for hah in range(grid[1]) ]
    h, w = float(HEIGHT/grid[1]), float(WIDTH/grid[0])
    
    for column in range(1,grid[1]+1):
        for row in range(1,grid[0]+1):
            fragment = src_hand[ceil((column-1)*h):min(ceil(column*h), HEIGHT),ceil((row-1)*w):min(ceil(row*w),WIDTH)]
            _,contour,_ = cv2.findContours(fragment,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            try: area = cv2.contourArea(contour[0])
            except: area=0.0
            area = float(area/(h*w))
            data[column-1][row-1] = area
    
    features = []
    for column in range(grid[1]):
        for row in range(grid[0]):
            features.append(data[column][row])
    return features


class HandMotionRecognizer:
    '''
    ### Hand Motion Recognizer class.

    This class is used to get motion information from each frame in a continuous feed. Use get_hand_motion() function to get motion information at each frame.
    '''
    def __init__(self):
        self.__prev_x = 0
        self.__prev_y = 0
        self.__threshold = 20

    def get_hand_motion(self, hand_contour):
        '''
        ### Get hand motion

        Inputs: Hand contour
        Returns: (-) If motion was found - Top/Left/Right/Down
                 (-) In case of no motion - False.
        '''
        
        import cv2

        M = cv2.moments(hand_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if self.__prev_x is 0 and self.__prev_y is 0: self.__prev_x, self.__prev_y = cx, cy

        delta_x, delta_y, slope = self.__prev_x-cx, self.__prev_y-cy, 0
        direction = 'None'

        if delta_x**2+delta_y**2 > self.__threshold**2:
            if delta_x is 0 and delta_y > 0: slope = 999 # inf
            elif delta_x is 0 and delta_y < 0: slope = -999 # -inf
            else: slope = float(delta_y/delta_x)
            
            if slope < 1.0 and slope >= -1.0 and delta_x > 0: direction = 'Right'
            elif slope < 1.0 and slope >= -1.0: direction = 'Left'
            elif (slope >= 1.0 or slope <=-1.0) and delta_y > 0.0: direction = 'Up'
            elif slope >= 1.0 or slope <=-1.0: direction = 'Down'
        
            self.__threshold = 7
            self.__prev_x, self.__prev_y = cx, cy

            return direction
        else:
            self.__threshold = 20
            return False
