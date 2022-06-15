import cv2, os
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical

test_data_dir = 'E:/MLHC-COVID-19/MLHC-COVID-19-DL/Dataset_holdout/test'

def load_test_data():
    labels = os.listdir(test_data_dir)
    X_test = []
    Y_test = []

    i = 0
    print('-' * 30)
    print('Creating test images...')
    print('-' * 30)
    for label in labels:
        i = 0
        image_names_test = os.listdir(os.path.join(test_data_dir, label))
        total = len(image_names_test)

############################################## Labeled for COVID-19 class ########################################################################
        if label == 'COVID-19' :
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

############################################## Labeled for Non-COVID-19 class ########################################################################
        elif label == 'Non-COVID':
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

############################################## Labeled for Healthy class ########################################################################
        elif label == 'Normal':
            j = 2
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

    le = preprocessing.LabelEncoder()
    Y_test = le.fit_transform(Y_test)
    Y_test = to_categorical(Y_test)
    ############################################# Save data to .NPY File ########################################################################
    # np.save('x_test_MLHC.npy', X_test)
    # np.save('y_test_MLHC.npy', Y_test)

    return X_test, Y_test

# X_test,y_test = load_test_data()  
