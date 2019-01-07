'''
# Step 1: Extracting features from image.

* Modify 'directory' to point to the directory containing the image.
* Modify 'labels' to insert all labels used by your model.
'''

import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'Indian-Sign-Language-Translator')))

import cv2

directory = 'data/'
labels = ['1','2','3','4','5','6','7','8','9','0']
TOTAL_IMAGES_PER_LABEL = 300
data_file_name = 'data.csv'

with open(data_file_name, 'w') as data:
    # Columns
    for i in range(100):
        data.write('f%d,' % i)
    data.write('label\n')

    for label in labels:
        for i in range(1, TOTAL_IMAGES_PER_LABEL+1):
            try:
                print('Processing image %d of label %s' % (i, label), end='\r')
                image = cv2.imread(directory+label+'/%d.png' % i)
                # Skin color segmentation
                segmented_image = utils.segment(image)

                # Face detection
                face_bounds, found_face = utils.detect_face(image)

                # Face elimination
                no_face_image = utils.eliminate_face(face_bounds, found_face, segmented_image)
                del segmented_image
                del image

                # Hand extraction
                hand = utils.get_my_hand(no_face_image)
                del no_face_image
                
                # Feature extraction
                feature_vector = utils.extract_features(hand)
                
                # Convert 'features' from list to str.
                feature_str = ''
                for feature_val in feature_vector:
                    feature_str += str(feature_val) + ','
                
                # Write to file
                data.write(feature_str+'%s\n' % label)
                
            except:
                continue
        print(' '*60, end='\r')

print(' '*60+'\rDone!')

start_training_model = input('Start training? [y/N]: ')
if start_training_model == 'y' or start_training_model == 'Y':
    from train import start_training
    start_training(data_file = data_file_name)