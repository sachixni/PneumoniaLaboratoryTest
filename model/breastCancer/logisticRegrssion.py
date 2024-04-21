import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob

# Provide image size
IMAGE_SIZE = [224, 224]

# Give our training and testing path
training_data = '../../breast_xray/train'
testing_data = '../../breast_xray/test'

# Find how many classes are present in the train dataset
folders = glob('../../breast_xray/train/*')

# Logistic Regression Model
input_shape = IMAGE_SIZE + [3]
model = tf.keras.Sequential([
    Flatten(input_shape=input_shape),
    Dense(len(folders), activation='softmax')  # Output layer with softmax for multiclass classification
])

# View the structure of the model
model.summary()

# Compiling our model with 'sparse_categorical_crossentropy' if the labels are integers
# Use 'categorical_crossentropy' if using one-hot encoded labels.
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

# ImageDataGenerators
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Set up the data generators
training_set = train_datagen.flow_from_directory(training_data,
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = 10,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(testing_data,
                                            target_size = IMAGE_SIZE,
                                            batch_size = 10,
                                            class_mode = 'categorical')

# Fitting the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5, # Change the number of epochs if needed
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# Save the model
model.save('logistic_regression_model.h5')
