import csv
import cv2
import numpy as np
import ntpath
import time
import pickle
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout


######################################################################################
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

################################################################################
### Function to read in image files and store them as pickle file
def read_csvs(datasets, steer_corr):
    ### Open CSV files of provided datasets
    images = []
    measurements = []
    for csvpath in datasets:
        with open(csvpath) as csvfile:
        # with open('data/Udacity/driving_log.csv') as csvfile:
            print('Processing file "',csvpath,'"... ',sep='',end='')
            reader = csv.reader(csvfile, skipinitialspace=True)
            j = 0
            for line in reader:
                if j == 0:
                    lines_header = line
                    j = j + 1
                    continue
                else:
                    # Update image file paths for center, left and right cam
                    line[0] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[0])[-1] # center
                    line[1] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[1])[-1] # left
                    line[2] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[2])[-1] # right
                    # Add image path information
                    images.extend([line[0],line[1],line[2]])
                    # Add steering angle information
                    measurements.extend([
                        float(line[3]),
                        float(line[3])+steer_corr,
                        float(line[3])-steer_corr])
            print('DONE!')
            print('  Total amount of datasets is now at',len(images),
                'and',len(measurements),'steering infos')
    return images, measurements
################################################################################
### Read in files
images, steer = read_csvs(datasets, steer_corr)
### Split datasets between "train" and "test" dataset
images_train, images_test, steer_train, steer_test = train_test_split(
                                                            images,
                                                            steer,
                                                            test_size=test_sz,
                                                            random_state=42)
################################################################################
### Keras Neural Network
# Print out
print("\nStarting Keras")
print("  'X_train' and 'y_train' with {} elements\n".format(len(images_train)))
print("  'X_test'  and 'y_test'  with {} elements\n".format(len(images_test)))
################################################################################
### Define generator function
# Generator is used to read in image files during batch creation. Reason for
# introducing this feature was memory limitations: model is now able to process
# as many image files as needed!
def generator(datasets, steer, batch_size):
    num_samples = len(datasets)
    while 1: # Loop forever so the generator never terminates
        datasets, steer = sklearn.utils.shuffle(datasets, steer)
        for offset in range(0, num_samples, int(batch_size/2)):
            # Half batch size is used, because images are augmented with
            # vertically rotated pictures --> multiplier of 2, but half as
            # many pictures should be read in one batch
            batch_datasets = datasets[offset:offset+int(batch_size/2)]
            batch_steers   = steer[offset:offset+int(batch_size/2)]

            images = []
            for batch_dataset in batch_datasets:
                image = cv2.cvtColor(cv2.imread(batch_dataset),cv2.COLOR_BGR2RGB)
                image_aug = cv2.flip(image,1)
                images.append(image)
                images.append(image_aug)
            angles = []
            for batch_steer in batch_steers:
                angles.extend([float(batch_steer),float(batch_steer)*-1])

            X_batch = np.array(images,dtype='float64')
            y_batch = np.array(angles,dtype='float64')

            yield sklearn.utils.shuffle(X_batch, y_batch)
################################################################################
### Create generators for train and test datasets
train_generator = generator(images_train, steer_train, batch_size=batch_sz)
test_generator = generator(images_test, steer_test, batch_size=batch_sz)
### Define model layout (based on NVidia model as indicated in video
# (URL: https://devblogs.nvidia.com/deep-learning-self-driving-cars/)
# Original NVidia input size: 66,200,3
model = Sequential()
# Perform normalization of image data
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
# Crop images: skip upper 70 and lower 25 pixels
model.add(Cropping2D(cropping=((70,25), (0,0))))  # remaining size: 65,320,3
# Several convolution layers, following NVidia
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(dropout_keep_rate))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
### Model optimization strategy
model.compile(loss='mse', optimizer='adam')
### Execute model
model.fit_generator(train_generator,
                    samples_per_epoch=len(images_train)*2, # x2: augmentation
                    nb_epoch=nb_epochs,
                    validation_data=test_generator,
                    nb_val_samples=len(images_test)*2) # x2: augmentation
### Save model
print("\n Saving neural network model as 'model.h5'...",end='')
model.save('model.h5')
print("DONE!")
print("Script finished.")
quit()
################################################################################
