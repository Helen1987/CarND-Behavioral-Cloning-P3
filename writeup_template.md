#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

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
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* image_processing.py includes image modification methods
* Training data.ipynb jupyter notebook to explore training data and create training and validation sets

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
**NB**. To run succesfully keras version 1.2.0 or higher, must be installed. Check [this issue](https://github.com/fchollet/keras/issues/4792) for details.

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

For this task I took pre-trained VGG16 model and use feature extraction approach (train only the top-level of the network, the rest of the network remains fixed). According to [c231n](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) normalization does not give any performance boost for images, so I skipped this step. Relu is used as activation layer.

####2. Attempts to reduce overfitting in the model

To smooth my car behavior I use L2 regularization. I use only one `Dropout` layer to prevent overvitting. It is enough for model to drive.
Car was trained on small balanced dataset from udacity data, a part from the rest data was used for validation set. On udacity data car drives backwards, so t1 is considered to be test track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86). Small L2 parameter (0.0001) fits better and standard 0.5 dropout is good enough.

####4. Appropriate training data

I took training data from udacity dataset (~400 nicely balanced images). 
![alt text][image1]
I use left and right images with +-25 adjustment to simulate recovery images. To avoid bias to left\right turns I use image flipping with changing angle sign. Brightness augmentation was used to generalize the model. To add more data for rare angles I add the same image with slighly modified angle (see `perturb_angle` from notebook)

###Model Architecture and Training Strategy

####1. Solution Design Approach

I select feature extraction approach for this task. The idea behind it. Models fro ImageNet competition are able to indendify the objects, so it must be able to identify the road successfully. So, I just need to tune it a little to force to stay in the middle of the road. VGG16 is quite small and provide good results, so I choose it like a base model.

First step was to get a balanced dataset. I divided data in bins with 0.1 step and choose the same number of images from each bin.

To prevent overfitting and smooze car driving I introcude L2 regularization, but it was not enough. I had to add dropout as well. One dropout showed the best result.

####2. Final Model Architecture

Frozen VGG16 with two Dense layers for regression task.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
