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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical




def get_features_song(f):
    y, _ = librosa.load(f)    
    mfcc = librosa.feature.mfcc(y)    
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)[:25000]

def create_features_and_labels():
    all_features = []
    all_labels = []

    genres = ['classical','hiphop','jazz','metal','pop','reggae']
    for genre in genres:
        sound_files = glob.glob('genres/'+genre+'/*.wav')
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in sound_files:
            features = get_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    new_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), new_labels



features, labels = create_features_and_labels()

print(np.shape(features))
print(np.shape(labels))

training_split = 0.8

alldata = np.column_stack((features, labels))           

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]

print(np.shape(train))
print(np.shape(test))

train_input = train[:,:-6]
train_labels = train[:,-6:]

test_input = test[:,:-6]
test_labels = test[:,-6:]

print(np.shape(train_input))
print(np.shape(train_labels))

#Create model
model = Sequential([Dense(100, input_dim=np.shape(train_input)[1]), Activation('relu'), Dense(6), Activation('softmax'),])

model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(train_input, train_labels, epochs=10, batch_size=32, validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Loss: %.4f, accuracy: %.4f" % (loss, acc))

model.save('my_model.h5')

