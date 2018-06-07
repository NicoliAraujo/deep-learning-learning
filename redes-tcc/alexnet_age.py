
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from models import AlexNet
from keras import backend as K
K.set_image_data_format('channels_last')


# In[2]:


train_dataset = BatchGenerator(box_output_format=['class_id'])
validation_dataset = BatchGenerator(box_output_format=['class_id'])

train_dataset.parse_csv(labels_filename='/home/nicoli/github/ssd_keras/dataset/csv/imdb_csv/imdb_age_regression_train_split_47950-70-10-20.csv', 
                        images_dir='../../ssd_keras/dataset/',
                        input_format=['image_name', 'class_id'])

validation_dataset.parse_csv(labels_filename='/home/nicoli/github/ssd_keras/dataset/csv/imdb_csv/imdb_age_regression_val_split_47950-70-10-20.csv', 
                             images_dir='../../ssd_keras/dataset/',
                             input_format=['image_name', 'class_id'])


# In[14]:


img_height, img_width, img_depth = (227,227,3)

epochs = 1

train_batch_size = 64
shuffle = True
ssd_train = False

validation_batch_size = 32


# In[15]:


train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         shuffle=shuffle,
                                         ssd_train=ssd_train,
                                         returns={'processed_labels'},
                                         resize=(img_height, img_width))

validation_generator = validation_dataset.generate(batch_size=validation_batch_size,
                                                   shuffle=shuffle,
                                                   ssd_train=ssd_train,
                                                   returns={'processed_labels'},
                                                   resize=(img_height, img_width))

print("Number of images in the dataset:", train_dataset.get_n_samples())
print("Number of images in the dataset:", validation_dataset.get_n_samples())


# In[16]:


steps_per_epoch = train_dataset.get_n_samples()/train_batch_size
validation_steps = validation_dataset.get_n_samples()/validation_batch_size


# In[6]:


alexnet = AlexNet(n_classes=1)


# In[7]:


alexnet.model.summary()


# In[8]:


optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)


# In[17]:


checkpoint = ModelCheckpoint(filepath='callbacks/alexnet/age/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=False,
                           period=5)
callbacks = [checkpoint]


# In[18]:


alexnet.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse', 'mae'])


# In[19]:


alexnet.model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)


# In[ ]:




