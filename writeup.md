#**Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model begins with two preprocessing layers. The first layer is a Lambda layer that normalize the image to mean zero and between -0.5 and 0.5. The second layer is a convolution layer with 1x1 filter of depth 3. This is used to give the model freedom to determine its own color space.

The following layers are NVIDIA arhitecture layers with drop out layers in between. Specifically, there are two convolutional layers: first three layers use 5x5 filter size with 2x2 strides, the next two layers use 3x3 filter size with 1x1 stride. The convolution layers have non-decreasing depths of 24, 36, 48, 64, and 64. The convolution layers are followed by fully connected layers of decreasing numbers of nodes: 100, 50, 10 and 1. 

I use ELU activation for both convolution layers and the hidden fully connected layers. No activation layer is used for the output layer since this is a regression problem and the targets steering angle.

#### 2. Attempts to reduce overfitting in the model

Initial attempts with training with extact NVIDIA architecture show that the model would overfit. I add dropout layers between layers to reduce overfitting. 

image of overfitting: 0.2

The model was trained and validated on different data sets to ensure that the model was not overfitting. The training data draw from the raw data and are processed by modifying brightness, vertical/horizontal shifts, or flipping left/right. The validation data are all the raw data taken with center camera.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. And it was also tested through the simulator on an unseen track to see if it could generalize to road conditions it never saw in the training process.

Visualization plot here

#### 3. Model parameter tuning

Learning rate is specified to be 0.0001 at the start of each training phase. During a phase, the learning rate was not tuned manually since it used an adam optimizer.

The probability to drop a node in each layer is a parameter that I tested by trial and error. I used the same rate for all the drop out layers. I tried 0., 0.2, 0.3, 0.4 and use the train/valid loss plot to choose the best one.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used images from center, left and right camera to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because the NVIDIA developed this model in their effort to train an end-to-end learning model for self-driving cars and achieved good performance.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that as training epochs increases, my first model has a monotonically decreasing mean squared error on the training set but the mse first decreases then increases on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that there's a drop out layer in between each convolution and fully connected layer.

Then I train the model with checkpoints storing weights of the model only when mse on the validation dataset reaches a new low compared to all the previous training epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or run out of the track when the side lanes are not obvious. To improve the driving behavior in these cases, I retrain the model with higher number of training images to increase the probability of images taken at those spots to occur in the training process.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

model table

| Layer Type | Output Shape | Params | Activation | 
|:----------:|:------------:|:------:|:----------:| 
| Lambda | ( , 66, 200, 3) | 12 | None |
| Convolution | ( , 31, 98, 24) | 1824 | ELU |
| Dropout | ( , 31, 98, 24) | 0 | None |
| Convolution | ( , 14, 47, 36) | 21636 | ELU |
| Dropout | ( , 14, 47, 36) | 0 | None |
| Convolution | ( , 5, 22, 48) | 43248 | ELU |
| Dropout | ( , 5, 2, 48) | 0 | None |
| Convolution | ( , 3, 20, 64) | 27712 | ELU |
| Dropout | ( , 1, 18, 64) | 0 | None |
| Flatten | ( , 1152 ) | 0 | None |
| Fully Connected | ( , 100 ) | 0 | ELU |
| Dropout | ( , 100) | 0 | None |
| Fully Connected | ( , 50 ) | 0 | ELU |
| Dropout | ( , 50) | 0 | None |
| Fully Connected | ( , 10 ) | 0 | ELU |
| Dropout | ( , 10) | 0 | None |
| Fully Connected | ( , 1 ) | 0 | ELU |


![alt text][image1]

#### 3. Training Dataset & Training Process

I used the dataset provided by the Udacity course website as my training dataset.

To augment the data sat, I applied three methods: brightness augmentation, translations and left/right flips.

For each training image, I convert the image into HSV space and apply a random multiplier between 0.7 to 1.3 to its V channel. Then I convert the image back into the RGB color space. Here are examples:

image for brightness

I also apply random translation along x and y axis by applying the cv2.warpAffine function with a translation Matrix: [[1, 0, tX], [0, 1, tY]]. The range I set up for horizontal shifts is smaller than 30% of total width both to the left and right. The range for verticle shifts is 6% of total height both to up and down. Along with any horizontal shifts, I add 0.004 to each pixel the image is shifted to the left. Here are examples:

images for shifts

Finally, I flipped images and angles thinking that this would remove any bias of a particular direction that the model would be trained from data with unbalanced steering angle. For example, here is an image that has then been flipped:

images for fliping

After the collection process, I use a Python generator so that I could have unlimited number of data points. I then preprocessed this data by cropping away the top 1/4 of the image to remove the distracting sky as well as the bottom 25 rows occupied by the front hood of the car. I then resize the image to match the input image dimension that NVIDIA model uses. Here's an example:

image for crop

I finally used all the images in the raw dataset as validation. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 for each phase as evidenced by the following train/valid loss plot.

images for two phases train/loss plot
I used an adam optimizer so that manually training the learning rate wasn't necessary during each phase of training.
