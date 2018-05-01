import csv
import cv2
import numpy as np
import ntpath
import time
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

######################################################################################
### Settings
use_datasets = [0,1,2]
nb_epochs = 1
batch_sz = 32
initialize_images = False
test_sz = 0.20

### List of available datasets
datasets = np.array([   'data/Udacity/driving_log.csv', \
                        'data/T1_Regular/driving_log.csv', \
                        'data/T1_OtherDir/driving_log.csv'], \
                        dtype='str')
datasets = datasets[use_datasets]

######################################################################################
### Function to read in image files and store them as pickle file
def read_images(datasets):
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
            print('DONE! Total amount of images sums up to',len(lines))

    ### Read images
    images = []
    measurements = []
    print("\nBegin reading of images:",)
    for line in tqdm(lines, total=len(lines)):
        source_path = line[0]
        # filename = source_path.split('/')[-1]
        # current_path = 'data/Udacity/IMG/' + filename
        image = cv2.imread(source_path)
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        measurement = float(line[3])
        measurements.append(measurement)
    print("Amount of steering angle information: ", len(measurements))
    print("Dimensions of image container:        ", np.shape(images))


    ### Split train and test data (with a fixed random seed)
    print("Splitting image data into train and test set ({}%)...".format(test_sz*100), end='')
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(images, dtype='uint8'),
        np.array(measurements, dtype='float32'),
        test_size=test_sz, random_state=42)
    print("DONE!")
    print(X_train.dtype, X_test.dtype, y_train.dtype, y_test.dtype)

    ### Write pickle file
    print("Saving image data in pickle file...", end='')
    f = open("data/train_dataset.pkl", "wb")
    pickle.dump([X_train, X_test, y_train, y_test], f)
    print("DONE!")
    return X_train, X_test, y_train, y_test

######################################################################################
### Function to augment available image files
def augment_images(X_samples, y_samples):
    X_augmented, y_augmented = [], []
    for image, measurement in zip(X_samples,y_samples):
        X_augmented.append(image)
        y_augmented.append(measurement)
        X_augmented.append(cv2.flip(image,1))
        y_augmented.append(measurement*-1.0)
    return np.array(X_augmented,dtype='uint8'), np.array(y_augmented,dtype='float32')

######################################################################################
### Main script
######################################################################################
### Read in image files (directly or pickled)
if initialize_images:
    print('\nReading in of datasets is being prepared (merge CSV files, generate pickled file)...')
    X_train, X_test, y_train, y_test = read_images(datasets)
else:
    print('\nOpening dataset from prepared pickle file...', end='')
    f = open("data/train_dataset.pkl", 'rb')
    X_train, X_test, y_train, y_test = pickle.load(f)
    print('DONE!')

### Augment image data
print("Augmenting image data (flip vertically)...",end='')
X_train, y_train = augment_images(X_train,y_train)
X_test, y_test = augment_images(X_test,y_test)
print(" DONE!")

### Keras Neural Network
print("\nStarting Keras instance with ")
print("  'X_train' ({}) and 'y_train' ({})\n".format(np.shape(X_train), len(y_train)))
print("  'X_test'  ({}) and 'y_test'  ({})\n".format(np.shape(X_test), len(y_test)))
# print("\nCreating video {}, FPS={}".format(args.image_folder, args.fps))

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
# from keras.preprocessing.image import ImageDataGenerator
import sklearn

### Define generator function
def generator(X_samples, y_samples, batch_size):
    # batch_size = int(batch_size * 0.5)
    num_samples = len(y_samples)
    while 1: # Loop forever so the generator never terminates
        X_samples, y_samples = sklearn.utils.shuffle(X_samples, y_samples)
        for offset in range(0, num_samples, batch_size):
            # y_batch = np.append(
            #     y_samples[offset:offset+batch_size]*-1,
            #     y_samples[offset:offset+batch_size],
            #     axis=0)
            # X_batch = np.append(
            #     X_samples[offset:offset+batch_size],
            #     X_samples[offset:offset+batch_size],
            #     axis=0)
            # for img in X_batch[0:batch_size]:
            #     img = cv2.flip(img,1)
            X_batch = X_samples[offset:offset+batch_size]
            y_batch = y_samples[offset:offset+batch_size]


            # images = []
            # angles = []
            # for batch_sample in batch_samples:
            #     name = './IMG/'+batch_sample[0].split('/')[-1]
            #     center_image = cv2.imread(name)
            #     center_angle = float(batch_sample[3])
            #     images.append(center_image)
            #     angles.append(center_angle)

            # TODO trim image to only see section with road
            # X_train = np.array(images)
            # y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_batch, y_batch)

train_generator = generator(X_train, y_train, batch_size=batch_sz)
test_generator = generator(X_test, y_test, batch_size=batch_sz)

#
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
# datagen.fit(X_train)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, validation_data=(X_test,y_test), shuffle=True, nb_epoch=nb_epochs)
# model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_sz),
#                     samples_per_epoch=len(X_train),
#                     nb_epoch=nb_epochs,
#                     validation_data=(X_test,y_test))
model.fit_generator(train_generator,
                    samples_per_epoch=len(X_train),
                    nb_epoch=nb_epochs,
                    validation_data=test_generator,
                    nb_val_samples=len(X_test))

print("\n Saving neural network model as 'model.h5'...",end='')
model.save('model.h5')
print("DONE!")
print("Script finished.")
