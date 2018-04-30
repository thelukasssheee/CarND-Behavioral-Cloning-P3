import csv
import cv2
import numpy as np

### Open CSV file of Udacity test image dataset
lines = []
with open('data/Udacity/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines_header = lines[0]
lines.remove(lines[0])
# print(lines[0])
# print(lines_header)
# lines.remove(lines[0])
# print(lines[0])
# quit()

### Read images
images = []
measurements = []
print("Begin reading of images...", end=" ")
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/Udacity/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
print("done!")
print("Steering angle information:", len(measurements))
print("Dimensions of image container:", np.shape(images))
#quit()

### Keras Neural Network
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
