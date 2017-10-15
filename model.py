import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2

path = './data/IMG/'
filenames = ['center_2016_12_01_13_33_45_217.jpg', 'center_2016_12_01_13_33_54_272.jpg', 'center_2016_12_01_13_33_59_244.jpg']

steer = [0.406227, -0.2306556, 0]
images = []
for file_name in filenames:
    file_path = path + file_name
    img = cv2.imread(file_path)   #cv2 read in images as BGR (not RGB)
    images.append(img)

# convert to numpy arrays for Keras
X_train = np.array(images)
y_train = np.array(steer)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, nb_epoch = 2) #, validation_split = 0.2, shuffle = True

model.save('model.h5')
