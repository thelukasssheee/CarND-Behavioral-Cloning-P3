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
nb_epochs = 3
batch_sz = 32
test_sz = 0.20
steer_corr = 0.00

### List of available datasets
datasets = np.array([   'data/Udacity/driving_log.csv', \
                        'data/T1_Regular/driving_log.csv', \
                        'data/T1_OtherDir/driving_log.csv'], \
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
                    line[0] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[0])[-1]
                    line[1] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[1])[-1]
                    line[2] = ntpath.split(csvpath)[0] + \
                                    '/IMG/'+ntpath.split(line[2])[-1]
                    images.extend([
                        line[0],
                        line[1],
                        line[2]])
                    measurements.extend([
                        float(line[3]),
                        float(line[3])+steer_corr,
                        float(line[3])-steer_corr])
            print('DONE!')
            print('  Total amount of information is now at',len(images),
                'and',len(measurements),'steering infos')
    return images, measurements
################################################################################

### Read in files and split train and test dataset
images, steer = read_csvs(datasets, steer_corr)
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

# Import functions
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D
import sklearn

################################################################################
### Define generator function
def generator(datasets, steer, batch_size):
    num_samples = len(datasets)
    while 1: # Loop forever so the generator never terminates
        datasets, steer = sklearn.utils.shuffle(datasets, steer)
        for offset in range(0, num_samples, int(batch_size/2)):
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

# Create generators for train and test datasets
train_generator = generator(images_train, steer_train, batch_size=batch_sz)
test_generator = generator(images_test, steer_test, batch_size=batch_sz)

# Define model layout
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))  # remaining: 65,320,3
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
# Define model optimization
model.compile(loss='mse', optimizer='adam')
# Execute model
model.fit_generator(train_generator,
                    samples_per_epoch=len(images_train)*2,
                    nb_epoch=nb_epochs,
                    validation_data=test_generator,
                    nb_val_samples=len(images_test)*2)
# Save model
print("\n Saving neural network model as 'model.h5'...",end='')
model.save('model.h5')
print("DONE!")
print("Script finished.")
quit()
################################################################################

    # ### Read images
    # images = []
    # measurements = []
    # print("\nBegin reading of images:",)
    # for line in tqdm(lines, total=len(lines)):
    #     source_center = line[0]
    #     source_left = line[1]
    #     source_right = line[2]
    #     images.extend(
    #         cv2.cvtColor(cv2.imread(source_center), cv2.COLOR_BGR2RGB),
    #         cv2.cvtColor(cv2.imread(source_left),   cv2.COLOR_BGR2RGB),
    #         cv2.cvtColor(cv2.imread(source_right),   cv2.COLOR_BGR2RGB),
    #         )
    #     measurement = float(line[3])
    #     measurement_left = measurement + steer_corr
    #     measurement_right = measurement - steer_corr
    #     measurements.extend(measurement,measurement_left,measurement_right)
    # print("Amount of steering angle information: ", len(measurements))
    # print("Dimensions of image container:        ", np.shape(images))
    #
    #
    # ### Split train and test data (with a fixed random seed)
    # print("Splitting image data into train and test set ({}%)...".format(test_sz*100), end='')
    # X_train, X_test, y_train, y_test = train_test_split(
    #     np.array(images, dtype='uint8'),
    #     np.array(measurements, dtype='float32'),
    #     test_size=test_sz, random_state=42)
    # print("DONE!")
    # print(X_train.dtype, X_test.dtype, y_train.dtype, y_test.dtype)
    #
    # ### Write pickle file
    # print("Saving image data in pickle file...", end='')
    # f = open("data/train_dataset.pkl", "wb")
    # pickle.dump([X_train, X_test, y_train, y_test], f)
    # print("DONE!")
    # return X_train, X_test, y_train, y_test
#
# ######################################################################################
# ### Function to augment available image files
# def augment_images(X_samples, y_samples):
#     X_augmented, y_augmented = [], []
#     for image, measurement in zip(X_samples,y_samples):
#         X_augmented.append(image)
#         y_augmented.append(measurement)
#         X_augmented.append(cv2.flip(image,1))
#         y_augmented.append(measurement*-1.0)
#     return np.array(X_augmented,dtype='uint8'), np.array(y_augmented,dtype='float32')

######################################################################################
### Main script
######################################################################################
### Read in image files (directly or pickled)
# if initialize_images:
#     print('\nReading in of datasets is being prepared (merge CSV files, generate pickled file)...')
#     X_train, X_test, y_train, y_test = read_images(datasets, steer_corr)
# else:
#     print('\nOpening dataset from prepared pickle file...', end='')
#     f = open("data/train_dataset.pkl", 'rb')
#     X_train, X_test, y_train, y_test = pickle.load(f)
#     print('DONE!')
#
# ### Augment image data
# print("Augmenting image data (flip vertically)...",end='')
# X_train, y_train = augment_images(X_train,y_train)
# X_test, y_test = augment_images(X_test,y_test)
# print(" DONE!")


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
