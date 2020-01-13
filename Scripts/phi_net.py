from keras.utils.vis_utils import plot_model
from keras import models
from keras import layers
from keras import optimizers
import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
import numpy as np
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten



#create model
model = Sequential()
#add model layers
model.add(Conv2D(16, kernel_size=3, activation='relu', input_shape=(128,128,1)))

model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(Conv2D(16, kernel_size=3, activation='relu'))

model.add(Flatten())
model.add(Dense(4, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()

#compile model using accuracy to measure model performance
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])


X = np.load('img_3d_array.npy')
X = np.expand_dims(X, axis=-1)

with open ('ref_file.txt', 'rb') as fp:
    y = pickle.load(fp)

y = pd.DataFrame({'ref':y})

y = y.replace(['?', 'SPINE_OTHER', 'SPINE_T1W', 'SPINE_T2W', 'SPINE_FLAIR'], 'OTHER')
y = y.astype('category', categories=["T1W","T2W","FLAIR","OTHER"])
y = pd.get_dummies(y)
y_np = y.values

X_train, X_test, y_train, y_test = train_test_split(X,y_np)




model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)