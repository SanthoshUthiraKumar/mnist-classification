# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model
![image](https://github.com/Pravinrajj/mnist-classification/assets/117917674/c8ed1138-8e1f-489b-96eb-a89cd5b81918)

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.
### STEP 2:
Download and load the dataset
### STEP 3:
Scale the dataset between it's min and max values
### STEP 4:
Using one hot encode, encode the categorical values
### STEP 5:
Split the data into train and test
### STEP 6:
Build the convolutional neural network model
### STEP 7:
Train the model with the training data
### STEP 8:
Plot the performance plot
### STEP 9:
Evaluate the model with the testing data
### STEP 10:
Fit the model and predict the single input


## PROGRAM

```
### Name: Santhosh U
### Register Number: 212222240092
```
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
     
(X_train, y_train), (X_test, y_test) = mnist.load_data()
     
X_train.shape
     
X_test.shape
     
single_image= X_train[0]
     
single_image.shape
     
plt.imshow(single_image,cmap='gray')
     
y_train.shape

X_train.min()
     
X_train.max()
     
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
     
X_train_scaled.min()
     
X_train_scaled.max()
     
y_train[0]
     
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
     
type(y_train_onehot)
     
y_train_onehot.shape
     
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
     
y_train_onehot[500]
     
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add (layers. Input (shape=(28,28,1)))
model.add (layers. Conv2D (filters=32, kernel_size=(7,7), activation='relu'))
model.add (layers. MaxPool2D (pool_size=(2,2)))
model.add (layers. Flatten())
model.add (layers. Dense (32, activation='relu'))
model.add (layers. Dense (16, activation='relu'))
model.add (layers. Dense (8, activation='relu'))
model.add (layers. Dense (10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('/content/DL_EX3.1.png')
type(img)
img = image.load_img('/content/DL_EX3.1.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Output1](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/f7f65a7b-a317-4a3b-8583-cec43edaf727)
![=Output2](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/f7a6f48e-a7b8-4781-85bc-d931ee8127ad)

### Classification Report
![Output4](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/05941935-e4f1-4fb0-9cbb-717708a50e3a)

### Confusion Matrix
![Output3](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/c90c5338-3872-4abf-a720-39529473d25c)

### New Sample Data Prediction
#### Input:
![DL_EX3](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/0472afe2-b543-4821-901e-297ff57f7292)

#### Output:
![Output5](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/41e9dbfc-7072-4a70-a30e-b55e4dee111f)
![Output6](https://github.com/SanthoshUthiraKumar/mnist-classification/assets/119477975/2127d829-76c4-45c1-99c0-6b85069158c7)


## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
