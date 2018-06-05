from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import warnings

class Conv2Net:
    def __init__(self, width, height, depth, classes, weights_path=None):
        model=Sequential()
        
        model.add(Conv2D(20, (5, 5) ,padding='same', input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Conv2D(50, (5, 5) , padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        if weights_path is not None:
            model.load_weights(weights_path)
            
        self.model = model
         

class Conv2NetRegressor:
    def __init__(self, width, height, depth, weights_path=None):
        model=Sequential()
        
        model.add(Conv2D(32, (3, 3) ,padding='same', input_shape=(height, width, depth)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        
              
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        
        model.add(Dense(1))
        model.add(Activation('relu'))
        
        if weights_path is not None:
            model.load_weights(weights_path)
            
        self.model = model

class AlexNet:
    def __init__(self, n_classes, weights_path=None):
        self.model = self.build_model(n_classes, weights_path)
        
    def build_model(self, n_classes, weights_path=None):
        if K.image_data_format()=='channels_first':
            input_shape=(3, 227,227)
        elif K.image_data_format()=='channels_last':
            input_shape=(227,227,3)
                
        model = Sequential()

        model.add(Conv2D(96, (11,11), strides=(4,4), input_shape=input_shape, padding='same', activation='relu', name='conv_1'))
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
            warnings.warn('Considering a regression task with output function being \'relu\'')
            model.add(Activation('relu', name='relu'))
        else:
            model.add(Activation('softmax', name='softmax'))
            
        if weights_path:
            model.load_weights(weights_path, by_name=True)

        return model