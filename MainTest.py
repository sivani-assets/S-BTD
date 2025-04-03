import cv2
import keras
from PIL import Image
import numpy as np


model=keras.models.load_model('BrainTumor10EpochsCategorical.keras')

image=cv2.imread('E:\III BCA\III BCA PROJECT\Brain Tumor Image Classification\pred\pred7.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result=np.argmax(model.predict(input_img),axis=1)

print(result)





