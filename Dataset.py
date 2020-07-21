import os, re, glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

groups_folder_path = 'algae2/'
categories = ['brown', 'green', 'red']

nClass = len(categories)

imgW = 224
imgH = 224

X = []
Y = []

for idx, categories in enumerate(categories):

	label = [0 for i in range(nClass)]
	label[idx] = 1
	imgDir = groups_folder_path + categories + '/'

	for top, dir, f in os.walk(imgDir):
		for filename in f:
			print(imgDir + filename)
			img = cv2.imread(imgDir + filename)
			img = cv2.resize(img, None, fx = imgW/img.shape[1], fy=imgH/img.shape[0])
			X.append(img/255)
			Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=608)
xy = (X_train, X_test, Y_train, Y_test)

np.save('algae2.npy', xy)

