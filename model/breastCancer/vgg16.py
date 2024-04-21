import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Provide image size
IMAGE_SIZE = [224, 224]

# Give our training and testing path
training_data = '../../breast_xray/train'
testing_data = '../../breast_xray/test'

# Import vgg16 model, 3 means working with RGB type of images. only two categories as Pneumonia and Normal
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Set the trainable as False, So that all the layers would not be trained.
# Use the pre-trained weights for classification. [transfer learning]
for layer in vgg.layers:
    layer.trainable = False

# Finding how many classes present in the train dataset.
folders = glob('../../breast_xray/train/*')
x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)
# Create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# View the structure of the model
model.summary()

# Compiling our model using adam optimizer and optimization metric as accuracy.
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialized for the image size
training_set = train_datagen.flow_from_directory(training_data,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(testing_data,
                                            target_size = IMAGE_SIZE,
                                            batch_size = 10,
                                            class_mode = 'categorical')

# Fitting the model.
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5, # You can change the number of epochs if you want
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# Save the model
model.save('vgg16.h5')