from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate, Input, Flatten, Dropout, Dense, Activation

#K.set_image_data_format('channels_first')
import warnings
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import SGD

class AlexNet:
    def __init__(self, n_classes, weights_path=None):
        
        self.model = self.build_model(n_classes, weights_path)
        
    def build_model(self, n_classes, weights_path=None):
        if K.image_data_format()=='channels_first':
            input_shape=(3, 227,227)
        elif K.image_data_format()=='channels_last':
            input_shape=(227,227,3)
                
        model = Sequential()

        model.add(Conv2D(96, (11,11), strides=(4,4), input_shape=(3, 224,224), padding='same', activation='relu', name='conv_1'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

        model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', activation='relu', name='conv_2'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='convpool_1'))

        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', name='conv_3'))
        
        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', name='conv_4'))
        
        model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', name='conv_5'))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='convpool_5'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(4096, activation='relu', name='dense_1'))
        model.add(Dense(4096, activation='relu', name='dense_2'))
        
        model.add(Dense(n_classes, name='dense_3'))
        
        if n_classes==1:
            waring.warn('Considering a regression task with output function being \'relu\'')
            model.add(Activation('relu', name='relu'))
        else:
            model.add(Activation('softmax', name='softmax'))
            
        if weights_path:
            model.load_weights(weights_path, by_name=True)

        return model
