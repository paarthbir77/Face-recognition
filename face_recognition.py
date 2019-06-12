#Importing required libraries
from keras.models import model_from_json
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, ZeroPadding2D, MaxPool2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import keras
import matplotlib.image as mpimg


import os
import tensorflow as tf
#To overcome 10% memory error on Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#keras.backend.set_image_data_format('channels_first')
# Defining Model VGG-Face
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=( 224,224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPool2D((2,2), strides=(2,2)))

model.add(Conv2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))
print("model made")


#model.load_weights('/home/paarth/Downloads/vgg-face-keras.h5')
model.load_weights('/home/paarth/Downloads/vgg_face_weights.h5')

print("model loaded")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

#Functions to find image distance
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    cos_sim = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    return cos_sim

def cosinesim(normalize_a, normalize_b):
    x = tf.constant(np.random.uniform(-1, 1, 10))
    y = tf.constant(np.random.uniform(-1, 1, 10))
    s = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)
    cosinedist = (tf.Session().run(s))
    return 1-cosinedist

def findEuclideanDistance(source, test):
    euclidean_distance = source - test
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

# Setting threshold for CosineSimilarity
thresh = 0.45


def verifyFace(img1, img2, str1, str2):
    img1_representation = vgg_face_descriptor.predict(preprocess_image(str1))[0, :]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(str2))[0, :]

    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    print("Cosine similarity: ", cosine_similarity)
    print("Euclidean distance: ", euclidean_distance)

    if (cosine_similarity < thresh):
        print("Yes, same person!")
    else:
        print("They are not same person!")

    print("-----------------------------------------")

import cv2 as cv
#img = cv.imread("/home/paarth/Downloads/fer_test/t7fer.jpg")

#Testing images
str1 = "/home/paarth/Downloads/rdj_data.jpg"
#str2 = "/home/paarth/Downloads/rdj_test.jpg"
str2 = "/home/paarth/Downloads/rdj_test2.jpeg"
#str1 = "/home/paarth/Downloads/paarth_test.jpg"
#str2 = "/home/paarth/Downloads/fer_test/t7fer.jpg"
#str1 = "/home/paarth/Downloads/katy_data.jpg"
#str1 = "/home/paarth/Downloads/katy_test.jpg"
#str2 = "/home/paarth/Downloads/katy_test2.jpg"
#str2 = "/home/paarth/Downloads/Zooey-Deschanel-Katy-Perry.jpg"
#str1 = "/home/paarth/Downloads/passport_photo_f622d3aa.jpeg"

img = cv.imread(str1)
img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA)
print("image 1 loaded")
#img = np.moveaxis(img, 2, 0)
imgset = cv.imread(str2)
imgset = cv.resize(imgset, (224, 224), interpolation=cv.INTER_AREA)
#imgset = np.moveaxis(imgset, 2, 0)
print("image 2 loaded")

cv.imshow("image1", img)
cv.imshow("image2", imgset)
verifyFace(img, imgset, str1, str2)

cv.waitKey(0)
cv.destroyAllWindows()
