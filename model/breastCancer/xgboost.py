import warnings
warnings.filterwarnings('ignore')
import numpy as np
import xgboost as xgb
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import train_test_split

# Provide image size
IMAGE_SIZE = [224, 224]

# Give our training and testing path
training_data = '../../breast_xray/train'
testing_data = '../../breast_xray/test'

# Import VGG16 model for feature extraction
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])

# We will not train the model again, we are using the pre-trained weights
base_model.trainable = False

# Function to preprocess and extract features from a directory of images
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Size depends on the output of the base_model
    labels = np.zeros(shape=(sample_count))
    generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=10,
        class_mode='categorical',
        shuffle=False)
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(preprocess_input(inputs_batch))
        features[i * 10 : (i + 1) * 10] = features_batch
        labels[i * 10 : (i + 1) * 10] = labels_batch
        i += 1
        if i * 10 >= sample_count:
            break
    return features, labels

# Extract features
train_features, train_labels = extract_features(training_data, 2000)  # Adjust the number of samples
test_features, test_labels = extract_features(testing_data, 1000)

# Flatten the features to fit into XGBoost
train_features = np.reshape(train_features, (2000, 7 * 7 * 512))
test_features = np.reshape(test_features, (1000, 7 * 7 * 512))

# Convert labels to 1D array
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1, random_state=42)

# Train XGBoost classifier
clf = xgb.XGBClassifier(objective='multi:softmax', num_class=len(glob('../../breast_xray/train/*')))
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_val, y_val)
print('Validation accuracy:', accuracy)
