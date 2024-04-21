# from keras.models import Model
# from keras.layers import Flatten,Dense
# from keras.applications.vgg16 import VGG16 #Import all the necessary modules
import matplotlib.pyplot as plot
import matplotlib.pyplot as plot
from glob import glob

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGESHAPE = [224, 224, 3]
vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)
training_data = '../chest_xray/train'
testing_data = '../chest_xray/test'

for each_layer in vgg_model.layers:
	each_layer.trainable = False
classes = glob('../chest_xray/train/*')
flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)
final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.summary()
final_model.compile(
loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']
)
# from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
								shear_range = 0.2,
								zoom_range = 0.2,
								horizontal_flip = True)
testing_datagen = ImageDataGenerator(rescale =1. / 255)
training_set = train_datagen.flow_from_directory('../chest_xray/train',
												target_size = (224, 224),
												batch_size = 4,
												class_mode = 'categorical')
test_set = testing_datagen.flow_from_directory('../chest_xray/test',
											target_size = (224, 224),
											batch_size = 4,
											class_mode = 'categorical')
# fitted_model = final_model.fit_generator( #Fitting the model.
# training_set,
# validation_data=test_set,
# epochs=20,
# steps_per_epoch=len(training_set),
# validation_steps=len(test_set)
# )
# Use this
fitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

final_model.save('20epoches_model.h5') #Saving the model file.

# plot.plot(fitted_model.history['loss'], label='training loss') #Plotting the accuracies
# plot.plot(fitted_model.history['val_loss'], label='validation loss')
# plot.legend()
# plot.show()
# plot.savefig('LossVal_loss')
# plot.plot(fitted_model.history['acc'], label='training accuracy')
# plot.plot(fitted_model.history['val_acc'], label='validation accuracy')
# plot.legend()
# plot.show()
# plot.savefig('AccVal_acc')

