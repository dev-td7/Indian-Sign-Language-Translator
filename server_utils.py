predictions = []
maxQueueSize = 15   # This is the max size of queue `predictions`
noOfSigns = 128     # This is the domain of the values present in the queue `predictions`
minModality = int(maxQueueSize/2)   # This is the minimum number of times a sign must be present in `predictions` to be declared as consistent

def addToQueue(pred):
    '''
    Adds the latest sign recognized to a queue of signs. This queue has maxlength: `maxQueueSize`

    Parameters
    ----------
    pred : This is the latest sign recognized by the classifier.
            This is of type number and the sign is in ASCII format

    '''
    global predictions, maxQueueSize, minModality, noOfSigns
    if len(predictions) == maxQueueSize:
        predictions = predictions[1:]
    predictions += [pred]
 
def getConsistentSign(hand_pose):
    '''
    From the queue of signs, this function returns the sign that has occured most frequently 
    with frequency > `minModality`. This is considered as the consistent sign.

    Returns
    -------
    number
        This is the modal value among the queue of signs.

    '''
    global predictions, maxQueueSize, minModality, noOfSigns
    addToQueue(hand_pose)
    modePrediction = -1
    countModality = minModality

    if len(predictions) == maxQueueSize:
        # countPredictions = [0]*noOfSigns
        countPredictions = {}

        for pred in predictions:
            if pred != -1:
                try:
                    countPredictions[pred]+=1
                except:
                    countPredictions[pred] = 1
        
        for i in countPredictions.keys():
            if countPredictions[i]>countModality:
                modePrediction = i
                countModality = countPredictions[i]
    
    return modePrediction

def displayTextOnWindow(windowName,textToDisplay,xOff=75,yOff=100,scaleOfText=2):
    '''
    This just displays the text provided on the cv2 window with WINDOW_NAME: `windowName`

    Parameters
    ----------
    windowName : This is WINDOW_NAME of the cv2 window on which the text will be displayed
    textToDisplay : This is the text to be displayed on the cv2 window

    '''
    import numpy as np, cv2
    signImage = np.zeros((200,400,1),np.uint8)
    print(textToDisplay)
    cv2.putText(signImage,textToDisplay,(xOff,yOff),cv2.FONT_HERSHEY_SIMPLEX,scaleOfText,(255,255,255),3,8);
    cv2.imshow(windowName,signImage);


