'''
# Step 2: Training the KNN classifier

* Modify the data_file variable and put the name of your csv file.
'''

def start_training(data_file):
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import pickle

    # Fetch data
    print('Fetching %s' % data_file, end='\r')
    data = pd.read_csv(data_file, dtype = {100: np.unicode_})
    print(' '*40+'\rTotal data parsed: %d' % len(data))

    # Split data into training and testing samples
    X = data[ ['f' + str(i) for i in range(100)] ].values
    Y = data['label'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=164)

    # Fit training data into model
    print('Fitting model...', end='\r')
    classifier = KNeighborsClassifier(n_neighbors = 3, algorithm = 'brute')
    classifier.fit(X_train, Y_train)

    # Test on testing data
    print('Testing...'+' '*10,end='\r')
    acc = classifier.score(X_test, Y_test)
    print('Accuracy: %.3f%%' % (acc * 100))

    # Print confusion matrix
    print('\nGetting confusion matrix..')
    preds = classifier.predict(X_test)
    confused = confusion_matrix(Y_test, preds)
    for row in confused:
        for elem in row:
            print(elem, end=',')
        
        print()

    # Fit all data into model and save it for further use.
    classifier = KNeighborsClassifier(n_neighbors = 3, algorithm = 'brute')
    classifier.fit(X, Y)
    pickle.dump(classifier, open('../Models/model.sav', 'wb'))
    print('Model saved')

if __name__ == "__main__":
    
    # Whatever your csv is, just write it's name here,
    data_file = 'digits.csv'
    start_training(data_file = data_file)