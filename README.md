[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

# Behavioral Cloning Project [P3]
### by Michael Berner, Student @ Udacity Self Driving Car Engineer Program

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Task / Problem description
*In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.*

*We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.* 

![Udacity_Simulator](./examples/Udacity_Simulator.png "Udacity Simulator")

*We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.*


**The goals / steps of this project are the following:**

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The full original project description is located at [Udacity's GitHub repository](https://github.com/udacity/CarND-Behavioral-Cloning-P3).

## Project Writeup & Rubric Points

In this section, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation. 


### Files Submitted & Code Quality

#### 1. Required files
*Submission includes all required files and can be used to run the simulator in autonomous mode*

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **README.md** summarizing the results (write_up.md was replaced with this readme file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my ``drive.py`` file, the car can be driven autonomously around the track by executing ``python drive.py model.h5``

For additional fun and challenge, I tweaked the ``drive.py``to run at 30 mph instead of 8 mph. With the provided neural network, the car is still able to remain on the tarmac.

However, it is necessary to activate the Anaconda development environment carnd-term1 to have all necessary Python functions available with the correct version. ``source activate carnd-term1``

#### 3. Submission code is usable and readable

The ``model.py`` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code is structured in such a way, that it can read the training data which I created, which are indicated by ``use_datasets = [0,1,2]``. If you want to run the code with the originally provided data by Udacity, you need to uncomment the line ``# use_datasets = [3]`` and comment the line ``use_datasets = [0,1,2]``.

```
### Settings
use_datasets = [0,1,2]  # Array to select which datasets to process
# use_datasets = [3]      # Array to select which datasets to process
nb_epochs = 7           # Number of epochs for neural network training
batch_sz = 32           # Batch size for neural network
test_sz = 0.20          # Fraction of images to use for test set
steer_corr = 0.00       # Steering correction value (left, right camera)
dropout_keep_rate = 0.7 # Dropout rate to keep neurons

### List of available datasets
datasets = np.array([   'data/Udacity/driving_log.csv', \
                        'data/T1_Regular/driving_log.csv', \
                        'data/T1_OtherDir/driving_log.csv', \
                        'data/driving_log.csv'], \
                        dtype='str')
datasets = datasets[use_datasets]
```

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network, which was described in a [blog by nVidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars). 

with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![Model_Visualization](./examples/placeholder.png "Model Visualization")


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Grayscaling][image2]

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
