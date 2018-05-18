from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense


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
         