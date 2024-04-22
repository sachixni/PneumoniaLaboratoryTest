import warnings

warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from glob import glob
from sklearn.model_selection import train_test_split

# Provide image size
IMAGE_SIZE = [224, 224]

# Paths to your data.
training_data = '../../breast_xray/train'
testing_data = '../../breast_xray/test'

# Initialize VGG16 with the pre-trained ImageNet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + [3])
base_model.trainable = False  # Freeze the layers

# Prepare ImageDataGenerator for feature extraction
datagen = ImageDataGenerator(rescale=1. / 255)


# Function to extract features from images
def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7 * 7 * 512))  # This shape will depend on the output of the base_model
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=IMAGE_SIZE,
        batch_size=10,
        class_mode='categorical',
        shuffle=False)

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = base_model.predict(preprocess_input(inputs_batch))
        features[i * 10: (i + 1) * 10] = features_batch.reshape((10, 7 * 7 * 512))
        labels[i * 10: (i + 1) * 10] = labels_batch
        i += 1
        if i * 10 >= sample_count:
            break

    return features, labels


# Number of images in each part of the dataset
num_train_samples = len(glob(training_data + '/*/*.jpg'))  # Update path and extension if needed
num_test_samples = len(glob(testing_data + '/*/*.jpg'))

# Extract features from the images
train_features, train_labels = extract_features(training_data, num_train_samples)
test_features, test_labels = extract_features(testing_data, num_test_samples)

# Flatten labels to a single value
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.1, random_state=42)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = knn.predict(X_val)

# Calculate the accuracy and print the classification report
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation accuracy with KNN: {accuracy}')
print(classification_report(y_val, y_pred))

# Optionally, save the trained KNN model using joblib or pickle
# import joblib
# joblib.dump(knn, 'knn_model.pkl')
