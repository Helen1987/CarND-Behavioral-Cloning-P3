
import os
import pickle
import numpy as np
import random
import matplotlib.image as mpimg
import cv2
from sklearn.utils import shuffle

from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from image_processing import preprocess_image, change_brightness

FOLDER_PATH = 'train_data'

def read_image(image_path):
    full_image_path = os.path.join(FOLDER_PATH, image_path.strip())
    image = mpimg.imread(full_image_path)
    return preprocess_image(image)

# use flipping to avoid bias to left\right turns
# use brightness augmentaion to generalize to t2
def generate_steering_angle(data, batch_size=64):
    X = []
    Y = []
    while True:
        data = shuffle(data)    
        for line in data:
            image = read_image(line['center'])
            angle = line['angle']
            image_brightened = change_brightness(image)
            X.append(image_brightened)
            Y.append(angle)
            
            flipped_image = cv2.flip(image, 1)
            flipped_image_brightened = change_brightness(flipped_image)
            X.append(flipped_image_brightened)
            Y.append(-angle)

            if len(X)>=batch_size:
                X, Y = shuffle(X, Y)
                yield np.array(X), np.array(Y) # (image, steering angle)
                X=[]
                Y=[]
            

def generate_validation(data):
    X = []
    Y = []
    while True:
        data = shuffle(data)
        for line in data:
            angle = line['angle']
            image = read_image(line['center'])

            X.append(image)
            Y.append(angle)
            yield np.array(X), np.array(Y) # (image, steering angle)

def create_model():
    # create the base pre-trained model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=[48, 75, 3])

    x = base_model.output
    x = Flatten()(x)

    # and a regression layer to predict steering angle
    x = Dense(1000, activation='relu', name='fc1', W_regularizer=l2(0.0001))(x)
    #x = Dropout(0.5)(x)
    x = Dense(250, activation='relu', name='fc2', W_regularizer=l2(0.0001))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)

    model = Model(input=base_model.input, output=predictions)
    # train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    #model = load_model("model.h5")
    return model

if __name__ == '__main__':
    model = create_model()

    training_pickle = 'train.p'
    with open(training_pickle, 'rb') as handle:
        driving_info = pickle.load(handle)

    validation_pickle = 'validation.p'
    with open(validation_pickle, 'rb') as handle:
        validation_info = pickle.load(handle)
    
    checkpoint = ModelCheckpoint(filepath='model-{epoch:02d}.h5')
    callback_list = [checkpoint]
    print("train size", len(driving_info))
    print("validation size", len(validation_info))

    # train the model on the new data for a few epochs
    model.fit_generator(
        generate_steering_angle(driving_info, batch_size=32),
        samples_per_epoch=768, nb_epoch=50,
        validation_data=generate_validation(validation_info),nb_val_samples=len(validation_info)/7,
        callbacks=callback_list)

    print("Saving model weights and configuration file.")

    model.save('model.h5')
    
    print("model is saved")