from keras.models import Sequential, Model, load_model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import cv2
import numpy as np

np.random.seed(608)

nClass = 3

def ConvBlock0(x, nFilters):

	x = Conv2D(filters=nFilters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	x = Conv2D(filters=int(nFilters * 0.25), kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	x = Conv2D(filters=nFilters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	return x


def ConvBlcok1(x, nFilters):

	x = Conv2D(filters=nFilters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	x = Conv2D(filters=int(nFilters * 0.25), kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	x = Conv2D(filters=nFilters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(alpha=0.1)(x)

	x = MaxPooling2D(pool_size=(2, 2))(x)

	return x


X_train, X_test, Y_train, Y_test = np.load('algae2.npy')


InputShape = (224, 224, 3)

ModelInput = Input(shape=InputShape)

x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=InputShape)(ModelInput)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

# 112 ====> 56
x = ConvBlock0(x, 32) # Going to Deeeeeeeeeeeeep!
x = ConvBlcok1(x, 32)

# 56 ====> 28
x = ConvBlock0(x, 64)
x = ConvBlcok1(x, 64)

# 28 ====> 14
x = ConvBlcok1(x, 128)

# Flatten
x = Flatten()(x)

# 25,088 ====> 4,096
x = Dense(units=4096, activation='relu', use_bias=True)(x)

# 4,096 ====> 1,024
x = Dense(units=1024, activation='relu', use_bias=True)(x)

# 1,024 ====> 256
x = Dense(units=256, activation='relu', use_bias=True)(x)

# Last 
ModelOutput = Dense(units=nClass, activation='softmax')(x)

model = Model(ModelInput, ModelOutput)

model.summary()

opt = Adam(learning_rate=0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, nb_epoch=200)

model.save('Test3.h5')
