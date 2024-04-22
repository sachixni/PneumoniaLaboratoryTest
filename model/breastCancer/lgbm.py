import warnings

from model.pneumonia.vgg16 import folders

warnings.filterwarnings('ignore')
import numpy as np
import lightgbm as lgb
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from glob import glob

# Provide image size
IMAGE_SIZE = [224, 224]

# Paths to your data.
training_data = '../../breast_xray/train'
testing_data = '../../breast_xray/test'

# Initialize VGG16 with the pre-trained ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])
base_model.trainable = False  # Freeze the layers

# Prepare ImageDataGenerator for feature extraction
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Prepare the data generators
train_generator = train_datagen.flow_from_directory(
    training_data,
    target_size=IMAGE_SIZE,
    batch_size=32,  # Adjust batch size accordingly
    class_mode='categorical',
    shuffle=False)

test_generator = test_datagen.flow_from_directory(
    testing_data,
    target_size=IMAGE_SIZE,
    batch_size=32,  # Adjust batch size accordingly
    class_mode='categorical',
    shuffle=False)

# Extract features
train_features = base_model.predict(train_generator, steps=len(train_generator), verbose=1)
test_features = base_model.predict(test_generator, steps=len(test_generator), verbose=1)

# Flatten the features to fit LightGBM
num_train_samples = len(train_generator.filenames)
num_test_samples = len(test_generator.filenames)
train_features_flattened = np.reshape(train_features, (num_train_samples, -1))
test_features_flattened = np.reshape(test_features, (num_test_samples, -1))

# Get the labels
train_labels = train_generator.classes
test_labels = test_generator.classes

# Create the LGBM datasets
train_data = lgb.Dataset(train_features_flattened, label=train_labels)
test_data = lgb.Dataset(test_features_flattened, label=test_labels)

# Define the parameters for the LGBM model
params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(folders),
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': -1,
    'min_child_samples': 100,
    'max_bin': 100,
    'subsample': 0.9,
    'subsample_freq': 1,
    'colsample_bytree': 0.8,
    'min_child_weight': 0.5,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0,
    'reg_lambda': 0,
    'nthread': -1,
    'verbose': 0,
    'is_unbalance': True
}

# Number of classes
num_class = len(folders)

# Train the LGBM model
lgbm_model = lgb.train(params, train_data, valid_sets=[train_data, test_data], num_boost_round=4000, early_stopping_rounds=100)

# Save the model
lgbm_model.save_model('lgbm_model.txt')
