import os, re, glob
import cv2
import numpy as np
import shutil
from numpy import argmax
from keras.models import load_model

categories = ['brown', 'green', 'red']

def PPImg(img_path):

	imgW = 224
	imgH = 224
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=imgW/img.shape[1], fy=imgH/img.shape[0])

	img = img/255

	return img


src = []
name = []
test = []
imgDir = 'algae2/green/'

for file in os.listdir(imgDir):

	if (file.find('.jpg') is not -1):

		src.append(imgDir + file)
		name.append(file)
		test.append(PPImg(imgDir + file))

test = np.array(test)
model = load_model('Test3.h5')

predict = model.predict(test)
predict = np.argmax(predict, axis=1)


for i in range(len(test)):
	print(name[i] + ' : , Predict : ' + str(categories[predict[i]]))
