from keras_preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

#Loading created model
model=load_model('pneumonia_model.h5')
img=image.load_img('../../static/Testing_Image_Folder/Test/test.jpeg', target_size=(224, 224))

#Converting the X-Ray into pixels
imagee=image.img_to_array(img)
imagee=np.expand_dims(imagee, axis=0)
# img_data=preprocess_input(imagee)
prediction=model.predict(imagee)

#Printing the prediction of model.
if prediction[0][0]>prediction[0][1]:
	print('Person is safe.')
else:
	print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')

if prediction[0][0]>prediction[0][1]:  #Printing the prediction of model.
    value =0
else:
    value =1

print(f'value: {value}')