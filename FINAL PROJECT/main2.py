# import libraries
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

# extract_features_song
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

# load model
new_model = tf.keras.models.load_model('my_model.h5')
##new_model.summary()
