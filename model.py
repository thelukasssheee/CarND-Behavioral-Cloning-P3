import csv
import cv2
import numpy as np
import ntpath
import time
from tqdm import tqdm

### Settings
use_datasets = [0,1]
nb_epochs = 4


### List of available datasets
datasets = np.array([   'data/Udacity/driving_log.csv', \
                        'data/T1_Regular/driving_log.csv'], \
                        dtype='str')
datasets = datasets[use_datasets]
print('\nReading in of datasets is being prepared (merge CSV files)...')

### Open CSV files of provided datasets
lines = []
for csvpath in datasets:
    with open(csvpath) as csvfile:
    # with open('data/Udacity/driving_log.csv') as csvfile:
        print('  Processing file "',csvpath,'"... ',sep='',end='')
        reader = csv.reader(csvfile, skipinitialspace=True)
        j = 0
        for line in reader:
            if j == 0:
                lines_header = line
                j = j + 1
                continue
            else:
                line[0] = ntpath.split(csvpath)[0]+'/IMG/'+ntpath.split(line[0])[-1]
                line[1] = ntpath.split(csvpath)[0]+'/IMG/'+ntpath.split(line[1])[-1]
                line[2] = ntpath.split(csvpath)[0]+'/IMG/'+ntpath.split(line[2])[-1]
                lines.append(line)
        print('done! Total amount of images sums up to',len(lines))

### Read images
images = []
measurements = []
print("\nBegin reading of images:",)
for line in tqdm(lines, total=len(lines)):
    source_path = line[0]
    # filename = source_path.split('/')[-1]
    # current_path = 'data/Udacity/IMG/' + filename
    image = cv2.imread(source_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print("Amount of steering angle information: ", len(measurements))
print("Dimensions of image container:        ", np.shape(images))

### Keras Neural Network
print("\n Starting Keras instance")
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=nb_epochs)

print("\n Saving NN model as 'model.h5'...",end='')
model.save('model.h5')
print("done!")
print("Script finished.")
