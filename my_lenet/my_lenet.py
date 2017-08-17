from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
import h5py
from keras.applications import imagenet_utils
from keras.models import model_from_json

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

#1)Load dataset
dataset = datasets.fetch_mldata("MNIST Original")
print(dataset.data[1].shape)
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
print(data[0].shape)
data = data[:, np.newaxis, :, :]
print(data.shape)
#2)Dataset partition int train and test data
#splits data into train and test by using test split of 33%
#using split method, and normalizes the pixel values
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data / 255.0, dataset.target.astype("int"), test_size=0.33)
print(trainData.shape, trainLabels, testData.shape, testLabels)
print(trainLabels)
# transform the training and testing labels into vectors in the
# range [0, classes] -- this generates a vector for each label,
# where the index of the label is set to `1` and all other entries
# to `0`; in the case of MNIST, there are 10 class labels
trainLabels = np_utils.to_categorical(trainLabels, 10)
print(trainLabels[0])
testLabels = np_utils.to_categorical(testLabels, 10)

#3)Loading model architecture
(width, height, depth, classes) = (28, 28, 1, 10)
print(width, height, depth, classes)
opt = SGD(lr=0.01) #optimizer

model = Sequential()#one layer after the other

# first set of CONV => RELU => POOL
model.add(Convolution2D(20, (5, 5), padding="same",
                        input_shape=(depth, height, width),
                        data_format="channels_first"))#you only have to declare data format on the first layer
model.add(Activation("relu"))#activation
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))#maxpooling--> 2x2 sliding window that walks a distance of 2 pixels to x and y

# second set of CONV => RELU => POOL
model.add(Convolution2D(50, (5, 5), padding="same")) #extracts 50 5x5 filters, same input
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# softmax classifier--> only in the end. Its the NN
model.add(Dense(classes))
model.add(Activation("softmax"))

#4)Compiling model
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

#5)Training model
model.fit(trainData, trainLabels, batch_size=128, nb_epoch=20,
          verbose=1)

(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=128, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
model = Sequential()#one layer after the other
model.save_weights("lenet_weights.hdf5", overwrite=True)

model_json = model.to_json()
print(model_json)
with open("my_model.json", "w+") as json_file:
    json_file.write(model_json)
