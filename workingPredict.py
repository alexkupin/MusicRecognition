'''
Created on Dec 11, 2019

@author: KConklin
'''

# Imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from telnetlib import theNULL

#Print version
print(tf.__version__)

def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

new_model = tf.keras.models.load_model('my_model.h5')

test = extract_features_song('classical.wav')
prediction = new_model.predict(np.array([test]))

np.set_printoptions(suppress=True)
print(prediction)

index = 0
largestIndex = 0;
largest = prediction[0][0]
for i in prediction[0]:
    if i > largest:
        largest = i
        largestIndex = index
    index = index + 1
    
if(largestIndex == 0):
    print("Song is classical!")
if(largestIndex == 1):
    print("Song is hiphop!")
if(largestIndex == 2):
    print("Song is jazz!")
if(largestIndex == 3):
    print("Song is metal!")
if(largestIndex == 4):
    print("Song is pop!")
if(largestIndex == 5):
    print("Song is reggae!")
    
    