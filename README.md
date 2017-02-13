# Project 3: Use Deep Learning to Clone Driving Behavior

[//]: # (Image References)

[train_data]: ./pic/train_set.png "Training dataset"
[generated_data]: ./pic/generated.png "Generated images"
[vgg16]: ./pic/VGG16.jpg "VGG16"

_The idea behind solution_

To clone car behavior I decided to use Feature extraction approach (train only the top-level of the network, the rest of the network remains fixed). Pre-trained network is able to identify different objects, so must be able to "find" the road. I just need to force the car to stay in the middle of it.
Some benefits of this approach:

1. small training dataset
2. reduce training time

My personal goal for this project was to try to generalize to t2 without training on t2 and with minimal amount of augmentaion. Unfortunately, my car is not able to generalize to t2.

_Architecture_

For a basic model with frozen features I select VGG16 since it is quite simple and shows good performance:

![alt text][vgg16]

After `Flatten` layer two more `Dense` layers (with 1000 and 250 neurons) were added on top for regression task with `Relu` as activation and L2 regularization with 0.0001 parameter. Then I have a standart 0.5 `Dropout` layer to prevent overfitting. And one more `Dense` layer for output value.

It is common to normalize your data near 0 to have a faster convergence, but according to [c231n](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC) normalization does not give any benefit when you work with images, so I skipped this step.
Since our task is pretty basic network architecture is not the key point for this task. Balanced and recovery data are more significant.

_Training_

The most important part for this project are data. I explore the udacity data set in various ways. Information is inside [Training data.ipynb](https://github.com/Helen1987/CarND-Behavioral-Cloning-P3/blob/master/Training%20data.ipynb). I wanted to get balanced data set. To add recovery data I use left and right images with +-0.25 angle adjustment. The top of the image is useless for our purpose, so I decided to remove it. I added flipped images to prevent bias to left and right curves.

To get a balanced dataset I divide my training set into bins with 0.01 step. From every bin I took the same pictures count. Overall I had ~400 images to train my car to drive.

To analyze my car behavior I saved the model after each iteration via callbacks. I noticed the car overfitted, so added L2 regularization. Initially, I added high parameter for L2 regularization as thought that my model suffers from high variance. After analyzing car behavior I realized the car suffers from high bias not from variance. As result, I decreased my L2 parameter to small value. Car started to drive smoothly but still my model overfitted. I added dropout before output layer. It helped and my car started to drive nicely. Initially, I struggled with the place after the bridge but then I realized that the simulator was in 'Fantastic' mode. I switched to the 'Fastest' mode and my car passed t1.

To generalize to track 2 I added brightness augmentaion to every image in generator. So, my car was trained on images (You can see the black region on image. I need small image size, so my computer is able to handle it. But VGG16 smallest input size is 48x48. I had to add region of interest technique instead of cropping to remove top of the image.):
![alt text][generated_data]

My model started to converge fast and car generalize to track 2 but was not able to pass it. I noticed that I have no enough data in my train set for >0.5 angles. As result, I generated some more data for rare angles (the same image but with slighly changed angle). Result train set:

![alt text][train_data]
