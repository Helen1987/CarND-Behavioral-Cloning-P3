#Behavioral Cloning

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* image_processing.py includes image modification methods
* Training data.ipynb jupyter notebook to explore training data and create training and validation sets
* record.mp4 video of track 1

#### 2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**NB**. To run succesfully keras version 1.2.0 or higher, must be installed. Check [this issue](https://github.com/fchollet/keras/issues/4792) for details.

#### 3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model arcthiecture has been employed

For this task I took pre-trained VGG16 model and use feature extraction approach (train only the top-level of the network, the rest of the network remains fixed). According to [c231n](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) normalization does not give any performance boost for images, so I skipped this step. Relu is used as activation layer.

#### 2. Attempts to reduce overfitting in the model

To smooth my car behavior I use L2 regularization. I use only one `Dropout` layer to prevent overvitting. It is enough for model to drive. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86). Small L2 parameter (0.0001) fits better and standard 0.5 dropout is good enough.

#### 4. Appropriate training data

Car was trained on small balanced dataset from udacity data, a part from the rest data was used for validation set. On udacity data car drives backwards, so t1 is considered to be test track.

I use left and right images with +-25 adjustment to simulate recovery images. To avoid bias to left\right turns I use image flipping with changing angle sign. Brightness augmentation was used to generalize the model. To add more data for rare angles I add the same image with slighly modified angle (see `perturb_angle` from notebook)


### Architecture and Training Documentation

See [readme](https://github.com/Helen1987/CarND-Behavioral-Cloning-P3/blob/master/README.md)

### Simulation

[t1 video](https://github.com/Helen1987/CarND-Behavioral-Cloning-P3/blob/master/record.mp4)

Car successfully passed t1. Honestly, right now I do not understand why it took so long. Now I am not able to create a NN which is not able to pass t1). 
