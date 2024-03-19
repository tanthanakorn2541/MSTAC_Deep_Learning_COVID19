import os
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class DataHandler:
    def __init__(self, train_data_dir, test_data_dir):
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
    
    def process_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((224, 224, 1))
        return img

    def load_data(self, directory, model_type):
        labels = os.listdir(directory)
        X = []
        Y = []

        for label in labels:
            image_names = os.listdir(os.path.join(directory, label))
            for image_name in image_names:
                if image_name != 'Thumbs.db':
                    img = self.process_image(os.path.join(directory, label, image_name))
                    
                    if model_type == 1:  # For the first model
                        if label == 'Normal':
                            Y.append(1)  # Normal to Healthy
                        else:
                            Y.append(0)  # COVID and Non-COVID to Unhealthy
                            
                    elif model_type == 2:  # For the second model
                        if label == 'COVID-19':
                            Y.append(0)  # COVID
                        else:
                            Y.append(1)  # Non-COVID

                    elif model_type == 3:  # For MSTAC model
                        if label == 'COVID-19':
                            Y.append(0)  # COVID
                        elif label == 'Non-COVID':
                            Y.append(1)  # Non-COVID
                        elif label == 'Normal':
                            Y.append(2)  # Normal
                            
                    X.append(img)

        X = np.array(X) / 255.0
        Y = to_categorical(Y)
        
        le = preprocessing.LabelEncoder()
        Y = le.fit_transform(Y)
        Y = to_categorical(Y)

        return X, Y

    def load_training_data(self, model_type, use_smote=False):
        X, Y = self.load_data(self.train_data_dir, model_type)
        
        if use_smote and model_type == 1:  # Apply SMOTE only for model type 1
            sm = SMOTE(random_state=42)
            X, Y = sm.fit_resample(X.reshape(X.shape[0], -1), Y)
            
            under = RandomUnderSampler(sampling_strategy=1)
            X, Y = under.fit_resample(X, Y)
            X = X.reshape(X.shape[0], 224, 224, 1)
            
            le = preprocessing.LabelEncoder()
            Y = le.fit_transform(Y)
            Y = to_categorical(Y)

        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        return X_train, X_val, y_train, y_val

    def load_testing_data(self, model_type):
        return self.load_data(self.test_data_dir, model_type)
