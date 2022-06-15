import cv2, os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.utils import to_categorical

train_data_dir = '../Dataset_holdout/train'
test_data_dir = '../Dataset_holdout/test'
############################################################ Load data #########################################################################      

def load_training_data():
    # Load training images
    labels = os.listdir(train_data_dir)
    total = len(labels)
    X_train = []
    Y_train = []

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    for label in labels:
        i = 0
        image_names_train = os.listdir(os.path.join(train_data_dir, label))
        total = len(image_names_train)

        ############################################## Labeled for Non-Healthy class ########################################################################
        if label == 'COVID-19' or label == 'Non-COVID':
            j = 0
            print(label,j)
            for image_name in image_names_train:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(train_data_dir, label, image_name)))
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.reshape((224,224,1))

                        X_train.append(img)
                        Y_train.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1

        ############################################# Labeled for Healthy class ########################################################################
        elif label == 'Normal':
            j = 1
            print(label,j)
            for image_name in image_names_train:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(train_data_dir, label, image_name)))
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.reshape((224,224,1))

                        X_train.append(img)
                        Y_train.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            
    print(i)
    print('Loading done.')

    X_train = np.array(X_train)/255.0
    Y_train = np.array(Y_train)


    ############################################# Balancing Data with SMOTE Technique ########################################################################
    sm = SMOTE(random_state=42)
    X_train, Y_train = sm.fit_resample(X_train.reshape(X_train.shape[0],-1), Y_train)
    
    under = RandomUnderSampler(sampling_strategy=1)
    X_train, Y_train = under.fit_resample(X_train,Y_train)
    X_train = X_train.reshape(X_train.shape[0],224,224,1)
    
    le = preprocessing.LabelEncoder()
    Y_train = le.fit_transform(Y_train)
    Y_train = to_categorical(Y_train)
    ############################################# Save data to .NPY File ########################################################################
    # np.save('x_train_layer1.npy', X_train)
    # np.save('y_train_layer1.npy', Y_train)

    return X_train, Y_train


def load_testing_data():
    # Load training images
    labels = os.listdir(test_data_dir)
    total = len(labels)
    X_test = []
    Y_test = []

    print('-' * 30)
    print('Creating testing images...')
    print('-' * 30)
    for label in labels:
        i = 0
        image_names_test = os.listdir(os.path.join(test_data_dir, label))
        total = len(image_names_test)

        ############################################## Labeled for Non-Healthy class ########################################################################
        if label == 'COVID-19' or label == 'Non-COVID':
            j = 0
            print(label,j)
            for image_name in image_names_test:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(test_data_dir, label, image_name)))
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.reshape((224,224,1))

                        X_test.append(img)
                        Y_test.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1

        ############################################# Labeled for Healthy class ########################################################################
        elif label == 'Normal':
            j = 1
            print(label,j)
            for image_name in image_names_test:
                try:
                    if image_name != 'Thumbs.db':
                        img = cv2.imread((os.path.join(test_data_dir, label, image_name)))
                        img = cv2.resize(img, (224, 224))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = img.reshape((224,224,1))

                        X_test.append(img)
                        Y_test.append(j)


                except Exception as e:
                    print(str(e))
                
                if i % 100 == 0:
                    print('Done: {0}/{1} images'.format(i, total))
                i += 1
            
    print(i)
    print('Loading done.')

    X_test = np.array(X_test)/255.0
    Y_test = np.array(Y_test)

    ############################################# Balancing Data with SMOTE Technique ########################################################################
    sm = SMOTE(random_state=42)
    X_test,Y_test = sm.fit_resample(X_test.reshape(X_test.shape[0],-1), Y_test)
    
    under = RandomUnderSampler(sampling_strategy=1)
    X_test, Y_test = under.fit_resample(X_test,Y_test)
    X_test = X_test.reshape(X_test.shape[0],224,224,1)
    
    le = preprocessing.LabelEncoder()
    Y_test = le.fit_transform(Y_test)
    Y_test = to_categorical(Y_test)
    

    ############################################# Save data to .NPY File ########################################################################
    # np.save('x_test_layer1.npy', X_test)
    # np.save('y_test_layer1.npy', Y_test)

    return X_test, Y_test

# X_train,y_train = load_training_data()
# X_test,y_test = load_testing_data()    
