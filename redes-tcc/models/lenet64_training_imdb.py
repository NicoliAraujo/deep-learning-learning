
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
from data_generator.batch_generator import BatchGenerator
from keras import optimizers
from keras.callbacks import *
from models import Conv2Net
from keras import backend as K
K.set_image_data_format('channels_last')


# In[13]:


train_dataset = BatchGenerator(box_output_format=['class_id1', 'class_id2'])
validation_dataset = BatchGenerator(box_output_format=['class_id1', 'class_id2'])

train_dataset.parse_csv(labels_filename='/home/nicoli/github/ssd_keras/dataset/csv/imdb_csv/imdb_gender_train_split_391087-70-10-2000.csv', 
                        images_dir='../../ssd_keras/dataset/',
                        input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id1', 'class_id2'])

validation_dataset.parse_csv(labels_filename='/home/nicoli/github/ssd_keras/dataset/csv/imdb_csv/imdb_gender_val_split_391087-70-10-2000.csv', 
                             images_dir='../../ssd_keras/dataset/',
                             input_format=['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id1', 'class_id2'])


# In[14]:


img_height, img_width, img_depth = (64,64,3)
classes = 2

epochs = 100

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


# In[18]:


conv2net = Conv2Net(img_width, img_height, img_depth, classes)


# In[19]:


conv2net.model.summary()


# In[20]:


optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=True)


# In[30]:


checkpoint = ModelCheckpoint(filepath='callbacks/conv2net/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                           monitor='val_loss',
                           verbose=1,
                           save_best_only=True,
                           period=5)
callbacks = [checkpoint]


# In[31]:


conv2net.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[32]:


conv2net.model.fit_generator(train_generator, epochs=epochs, 
                             steps_per_epoch=steps_per_epoch, 
                             validation_data=validation_generator,
                             validation_steps=validation_steps,
                             callbacks=callbacks)


# In[ ]:




