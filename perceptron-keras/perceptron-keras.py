from keras.models import Sequential
model = Sequential() #stack of nn layers
model.add(Dense(12, #12 neurons in one single layer
                input_dim=8, #eight input values
                kernel_initializer='random_uniform'))
