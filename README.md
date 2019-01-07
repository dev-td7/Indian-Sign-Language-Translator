# SiLaTra - The Indian Sign Language Translator

## Developers
* [Tejas Dastane](https://github.com/dev-td7)
* [Varun Rao](https://github.com/vrr-21)
* [Kartik Shenoy](https://github.com/kartik2112)
* [Devendra Vyavaharkar](https://github.com/DevendraVyavaharkar)

## Introduction

This is an Indian Sign Language Translator API. This API is capable of serving the following purposes:

* Hand Pose Recognition [Digits 0-9 and Letters A-Z, as well as other intermediate poses used in gesture]
* Gesture Recognition

SiLaTra API works only on one-handed gestures made with right hand. The sign performer **must** wear a Full sleeve T-Shirt of any colour other than near-skin and skin colours. See constraints below for a detailed account.

**This API can be extended to other sign languages as well!**
Just add your modules to recognise hand poses in the target sign language and corresponding HMM models for each gesture, and replace the model names used in `utils.py` and `silatra.py`. Scroll below for instructions on how to train your own custom model.

This API makes use of the following modules:

* Face recognition
* Skin Segmentation
* Hand region segmentation (Single Hand only)
* Object stabilisation (using facial reference)
* Hand motion detection

Following Gestures are supported:

1. Good Afternoon
2. Good Morning
3. Good Night
4. After
5. Leader
6. Please give me your pen
7. Strike
8. Apple
9. Towards
10. I am sorry
11. All the best
12. That is good


### External Links
* SiLaTra is actually built as an Android application which can recognize gestures and hand poses in real-time. The actual server-side implementation of this application can be found here - [SiLaTra server-side](https://github.com/kartik2112/Silatra). 
* The link to the Android application, which sends real-time feed to the server can be found here - [The Android application](https://github.com/DevendraVyavaharkar/Silatra-UDP).

## Usage

### Installing dependencies.
To install all dependencies, run `dependencies.sh` file. It will install all dependencies on it's own. This API is only made for Python 3.

### Ambient Lighting check.
Since recognition is completely dependent on Skin segmentation, ambient light conditions are desired. You can check this yourself by running `check_segmentation.py` file.

### Using the API functions
For Hand pose recognition and Gesture Recognition, import the file `silatra.py` file in your python script. This file contains the following:

* `recognise_hand_pose` function: For recognising Hand poses.
    * For Indian Sign Language Digits and Letters, directly use the function.
    * For classification of intermediate gesture poses (those with prefix `gesture_pose_` in ./samples) use parameter `model_path='Models/silatra_gesture_signs.sav'`
* Gesture class: Create an instance of this class. Use the following functions:
    * `add_frame`: Pass each frame of the unknown gesture you wish to classify into this function
    * `classify_gesture`: After all frames are passed, use this function to classify gesture and get the result.

### Samples

For usage of this API, two samples are available. `sample1.py` is an example of Hand pose recognition. `sample2.py` is an example of Gesture Recognition.

## Constraints

* Requires a background with no near skin colours.
* One hand only. The other hand **should not be visible**
* Sufficient amount of light is needed. Not too much not too less. And preferably avoid using in sunlight, as its yellow light will make everything look skin colour alike.
* The person performing gestures should wear a Full sleeve shirt. Only hand region should be visible, not your arm!

## Training your own model
You can create your own custom model to recognize hand poses. 

* Place your images into a directory in the `train` folder.
* Modify the directory path and labels in `train/process_images.py`.
* `train/process_images.py` will convert your images into a .csv file.
* Modify the dataset name in the `train/train.py` file and run the script. It will create a classifier from the .csv file and save it as a .sav file.
* You can find your model in the `./Models/` directory.

___

That's it.

_PS: This is our Bachelor's research project._