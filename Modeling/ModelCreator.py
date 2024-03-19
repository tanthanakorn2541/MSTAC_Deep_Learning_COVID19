import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, Dropout, Dense, Flatten
from tensorflow.keras.models import Sequential

class ModelCreator:
    @staticmethod
    def create_model():
        model = Sequential()
        model.add(BatchNormalization(input_shape=(224, 224, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.35))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.35))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.35))
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
