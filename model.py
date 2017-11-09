import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import random
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint  

# define a function to augment images by change its brightness
def random_brightness(img, factor):

    # assume input (img) is color image
    # convert Red, Green, Blue to Hue, Saturation, Value color space
    # factor describes the range of brightness scaling: (1 - factor, 1 + factor)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    bright_factor = random.uniform(1. - factor, 1. + factor)
    img_hsv = np.array(img_hsv, np.float)
    # set to 255 if pixel value is higher than 255 after multiplication
    img_hsv[:,:,2] = np.minimum(255, img_hsv[:,:,2] * bright_factor)
    img_hsv = np.array(img_hsv, dtype = np.uint8)
    img_new = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    
    return img_new

# define a function to augment images by shift it up/down, left/right
def xy_translation(img, steer, dfactor):

    # assume input (img) is color image
    # dfactor is the fraction of total pixels to shift left/right
    nrow, ncol, nch = img.shape
    tX = dfactor * ncol * random.uniform(-1., 1.)
    # give vertical translation less freedom, 1/5 of horizontal range
    tY = dfactor * 0.2 * nrow * random.uniform(-1., 1.) 
    Mtrans = np.float32([[1,0,tX], [0,1,tY]])
    img_new = cv2.warpAffine(img, Mtrans,(ncol, nrow))
    # adjust the steering angle by adding 0.004 every pixel shifted to the right
    steer_new = steer + tX * .004

    return img_new, steer_new

# define a function to crop images and then resize
def crop_resize(img, new_row, new_col):

    nrow, ncol, nch = img.shape
    # crop out img[y1:y2, x1:x2]
    # y1 = 0.3 * nrow: leave out the top 1/4 of image
    # y2 = nrow - 25: leave out the bottom 25 rows
    img_new = img[int(nrow * 0.25) : nrow - 25, 0 : ncol]
    img_new = cv2.resize(img_new,(new_col,new_row), \
                         interpolation=cv2.INTER_AREA)

    return img_new

# define a function to take a given image from data and augement it
def preprocess_train(entry):

    # entry is assumed to be a Series object with path to images, steering etc.
    # random with equal probability choose from center, left and right image
    icamera = np.random.randint(3)
    cam_pos = list(steer_correction.keys())[icamera]
    filename = entry[cam_pos].strip()
    # steer_correction is a dictionary w/ angle correction for l/r camera
    steering = float(entry['steering']) + float(steer_correction[cam_pos])
    img = cv2.imread(data_dir + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    br_img = random_brightness(img, 0.3)
    tr_img, steering = xy_translation(br_img, steering, 0.3)
    cr_img = crop_resize(tr_img, new_row, new_col)
    # random with equal probability flip or not flip an image
    flip_ornot = np.random.randint(2)
    if flip_ornot == 1:
        image = cv2.flip(cr_img, 1)    #1 means flip around y-axis
        steering = -steering
    else:
        image = cr_img
        
    image = np.array(image)    # convert to numpy arrays for Keras
    
    return image, steering

# define a function to take a given data entry and prepare it for validation
def preprocess_valid(entry):

    # entry is assumed to be a Series object with path to images, steering etc.
    # only use the image from center camera for validation
    filename = entry['center'].strip()
    steering = float(entry['steering'])
    img = cv2.imread(data_dir + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cr_img = crop_resize(img, new_row, new_col)
    image = np.array(cr_img)

    return image, steering

# define a generator to generate training data in batches
def generate_train_batch(log, batch_size = 128, filter_angle = 0.1, filter_ratio = 0.9):

    # batch_images is stored as float data type
    batch_images = np.zeros((batch_size, new_row, new_col, 3), dtype = float)
    batch_steering = np.zeros(batch_size, dtype = float)
    
    while 1:
        for i_batch in range(batch_size):
            
            # select images randomly from all entries
            # reselect with probability of filter_ratio if ...
            # steering angle is smaller than filter_angle
            reselect = 1
            while reselect == 1:
                i_entry = np.random.randint(len(log))
                entry = log.iloc[i_entry, :]
                image, steering = preprocess_train(entry)
                if abs(steering) < filter_angle:
                    tmp_rand = np.random.uniform(0., 1.)
                    if tmp_rand < filter_ratio:
                        reselect = 1
                    else:
                        reselect = 0
                else:
                    reselect = 0

            batch_images[i_batch] = image
            batch_steering[i_batch] = steering

        yield batch_images, batch_steering

# define a generator to generate training data in batches
def generate_valid(log):

    while 1:
        for i_entry in range(len(log)):
            entry = log.iloc[i_entry, :]
            image, steering = preprocess_valid(entry)
            image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            # note the output format
            steering = np.array([[steering]])

            yield image, steering

# build a new model using NVIDIA architecture
# add dropout layers in between layers to prevent overfitting
def build_model(drop_prob = 0.1, show_model = False):
    model = Sequential()
    model.add(Lambda(lambda x : x/255. - 0.5, input_shape = (new_row, new_col, 3)))
    # 3 @ 1x1 filter to choose color space automatically
    model.add(Convolution2D(3, 1, 1, init = 'he_normal', activation = 'elu', border_mode = 'valid', name = 'conv0'))
    model.add(Convolution2D(24, 5, 5, init = 'he_normal', subsample = (2, 2), border_mode = 'valid', activation = 'elu', name = 'conv1'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(36, 5, 5, init = 'he_normal', subsample = (2, 2), border_mode = 'valid', activation = 'elu', name = 'conv2'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(48, 5, 5, init = 'he_normal', subsample = (2, 2), border_mode = 'valid', activation = 'elu', name = 'conv3'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample = (1, 1), border_mode = 'valid', activation = 'elu', name = 'conv4'))
    model.add(Dropout(drop_prob))
    model.add(Convolution2D(64, 3, 3, init = 'he_normal', subsample = (1, 1), border_mode = 'valid', activation = 'elu', name = 'conv5'))
    model.add(Flatten())
    model.add(Dense(100, init = 'he_normal', activation = 'elu', name = 'full1'))
    model.add(Dropout(drop_prob))
    model.add(Dense(50, init = 'he_normal', activation = 'elu', name = 'full2'))
    model.add(Dropout(drop_prob))
    model.add(Dense(10, init = 'he_normal', activation = 'elu', name = 'full3'))
    model.add(Dropout(drop_prob))
    model.add(Dense(1, init = 'he_normal', name = 'full4'))

    if show_model:
        with open('report.txt','w') as f:
            model.summary(print_fn = lambda x: f.write(x + '\n'))

    return model


# data directory
data_dir = "./data/"
# read in driving log
driving_log = pd.read_csv(data_dir + "driving_log.csv")
n_entries, n_feats = driving_log.shape
# steering angle correction: add 0.2 to angle if the image is from the left camera
# positive angle means right turn, negative left turn
steer_correction = {'left': 0.2, 'center' : 0., 'right' : -0.2}

# convert images to new shape as is used in the NVIDIA architecture
new_row = 66
new_col = 200

# use a drop probability of 20%
drop_prob = 0.2

# total number of phases to train model
n_phases = 2

# i is the phase counter
i = 0

# train_cont = True means the training phase is continued from a previous model
train_cont = [False, True]
# number of epochs for each phase
epochs = [15, 10]
# train with 50% of images with angle < filter_angle for the first phase, 30% the second
filter_ratio = [0.5, 0.3]
batch_size = 128

# a ModelCheckpoint is used to save weights of the model only when validation loss improves 
checkpointer = ModelCheckpoint(filepath='weights.best.hdf5', 
                               verbose=1, save_best_only=True)

while i < n_phases:
    
    model = build_model(drop_prob)
    # use Adam optimizer with initial learning rate = 0.0001
    adam = Adam(lr=1e-4)
    # compile mode using loss function of mean squared error
    model.compile(loss = 'mse', optimizer = adam)
    # for phases after the first training phase, load the best trained model so far
    if train_cont[i]:
        model.load_weights('weights.best.hdf5')
    
    # generator to generate batches of training data
    train_generator = generate_train_batch(driving_log, batch_size = batch_size, filter_angle = 0.1, filter_ratio = filter_ratio[i])
    # valid image generator
    valid_generator = generate_valid(driving_log)

    # number of training images is 16,384 and validation data is all of the raw data from center camera
    history = model.fit_generator(train_generator,
            samples_per_epoch = 16384, 
            nb_epoch = epochs[i],
            validation_data=valid_generator,
            nb_val_samples=n_entries, 
            verbose = 1, 
            callbacks=[checkpointer])

    # save to file the training and validation loss for visualization
    np.savetxt('history_val_loss_'+str(i)+'.txt', history.history['val_loss'])
    np.savetxt('history_loss_'+str(i)+'.txt', history.history['loss'])

    i += 1


# convert the saved weights of the model to save model for autonomous driving
model.load_weights('weights.best.hdf5')
model.save('model.h5')
