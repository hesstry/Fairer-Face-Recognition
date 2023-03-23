# -*- coding: utf-8 -*-
"""Fairfacethings.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uQpZgdHyWtKR_5Om0EsBMaUp80aq15jk
"""

# UCSB ECE 281 CV Project\
# Much of the framework for this code came from the following github repo
# https://github.com/serengil/tensorflow-101/blob/master/python/Race-Ethnicity-Prediction-Batch.ipynb

from google.colab import drive
drive.mount('/content/drive')

import zipfile
!unzip /content/drive/MyDrive/fairfacefolder/fairface-img-margin025-trainval.zip

!unzip /content/drive/MyDrive/fairfacefolder/imgs_jpg.zip

!unzip /content/drive/MyDrive/fairfacefolder/vgg_face_weights.zip

!unzip /content/drive/MyDrive/fairfacefolder/fairface_label_train_w_synth.zip
!unzip /content/drive/MyDrive/fairfacefolder/fairface_label_val.zip

"""# New Section"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import keras.utils as image
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
tf.test.gpu_device_name()

tqdm.pandas()

"""Select which type of data to use. """

#Read the labels from the train and validation sets
train_df = pd.read_csv("fairface_label_train_w_synth.csv")
test_df = pd.read_csv("fairface_label_val.csv", nrows = 5000)
#controls the distribution of each race in the training set for real and synthetic _s
all_races = {'White' : 3000,'Black': 3000,'Latino_Hispanic': 3000,'Southeast Asian': 2000,'East Asian': 2000,'Indian': 3000,'Middle Eastern': 3000,
             'S_White' : 0,'S_Black': 0,'S_Latino_Hispanic': 0,'S_Southeast Asian': 0,'S_East Asian': 0,'S_Indian': 0,'S_Middle Eastern': 0}

# all_races_test = {'White' : 1000,'Black': 1000,'Latino_Hispanic': 1000,'Southeast Asian': 500,'East Asian': 500,'Indian': 1000,'Middle Eastern': 1000}
# Real White
white_a = pd.read_csv("fairface_label_train_w_synth.csv")
white_a.drop(train_df.index[(train_df['race'] != 'White')],axis=0,inplace=True)
white_a = white_a.reset_index(drop = True)
white_a = white_a.drop(labels = range(all_races['White'],white_a.shape[0]))

#Synthetic White
white_s = pd.read_csv("fairface_label_train_w_synth.csv")
white_s = white_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
white_s.drop(white_s.index[(white_s['race'] != 'White')],axis=0, inplace = True)
white_s = white_s.reset_index(drop = True)
white_s = white_a.drop(labels = range(all_races['S_White'],white_a.shape[0]))


# Real Black
black_a = pd.read_csv("fairface_label_train_w_synth.csv")
black_a.drop(train_df.index[(train_df['race'] != 'Black')],axis=0,inplace=True)
black_a = black_a.reset_index(drop = True)
black_a = black_a.drop(labels = range(all_races['Black'],black_a.shape[0]))

# Synthetic Black
black_s = pd.read_csv("fairface_label_train_w_synth.csv")
black_s = black_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
black_s.drop(black_s.index[(black_s['race'] != 'Black')],axis=0, inplace = True)
black_s = black_s.reset_index(drop = True)
black_s = black_a.drop(labels = range(all_races['S_Black'],black_a.shape[0]))

# Real Latino
latino_a = pd.read_csv("fairface_label_train_w_synth.csv")
latino_a.drop(train_df.index[(train_df['race'] != 'Latino_Hispanic')],axis=0,inplace=True)
latino_a = latino_a.reset_index(drop = True)
latino_a = latino_a.drop(labels = range(all_races['Latino_Hispanic'],latino_a.shape[0]))

# Synthetic Latino
latino_s = pd.read_csv("fairface_label_train_w_synth.csv")
latino_s = latino_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
latino_s.drop(latino_s.index[(latino_s['race'] != 'Latino_Hispanic')],axis=0, inplace = True)
latino_s = latino_s.reset_index(drop = True)
latino_s = latino_a.drop(labels = range(all_races['S_Latino_Hispanic'],latino_a.shape[0]))

# Real Southeast
south_a = pd.read_csv("fairface_label_train_w_synth.csv")
south_a.drop(train_df.index[(train_df['race'] != 'Southeast Asian')],axis=0,inplace=True)
south_a = south_a.reset_index(drop = True)
south_a = south_a.drop(labels = range(all_races['Southeast Asian'],south_a.shape[0]))

#Synthetic Southeat
south_s = pd.read_csv("fairface_label_train_w_synth.csv")
south_s = south_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
south_s.drop(south_s.index[(south_s['race'] != 'Southeast Asian')],axis=0, inplace = True)
south_s = south_s.reset_index(drop = True)
south_s = south_a.drop(labels = range(all_races['S_Southeast Asian'],south_a.shape[0]))

# Real East
east_a = pd.read_csv("fairface_label_train_w_synth.csv")
east_a.drop(train_df.index[(train_df['race'] != 'East Asian')],axis=0,inplace=True)
east_a = east_a.reset_index(drop = True)
east_a = east_a.drop(labels = range(all_races['East Asian'],east_a.shape[0]))

# Synthetic East
east_s = pd.read_csv("fairface_label_train_w_synth.csv")
east_s = east_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
east_s.drop(east_s.index[(east_s['race'] != 'East Asian')],axis=0, inplace = True)
east_s = east_s.reset_index(drop = True)
east_s = east_a.drop(labels = range(all_races['S_East Asian'],east_a.shape[0]))

#Real Indian
indian_a = pd.read_csv("fairface_label_train_w_synth.csv")
indian_a.drop(train_df.index[(train_df['race'] != 'Indian')],axis=0,inplace=True)
indian_a = indian_a.reset_index(drop = True)
indian_a = indian_a.drop(labels = range(all_races['Indian'],indian_a.shape[0]))

#Synthetic Indian
indian_s = pd.read_csv("fairface_label_train_w_synth.csv")
indian_s = indian_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
indian_s.drop(indian_s.index[(indian_s['race'] != 'Indian')],axis=0, inplace = True)
indian_s = indian_s.reset_index(drop = True)
indian_s = indian_a.drop(labels = range(all_races['S_Indian'],indian_a.shape[0]))

#Real Middle
middle_a = pd.read_csv("fairface_label_train_w_synth.csv")
middle_a.drop(train_df.index[(train_df['race'] != 'Middle Eastern')],axis=0,inplace=True)
middle_a = middle_a.reset_index(drop = True)
middle_a = middle_a.drop(labels = range(all_races['Middle Eastern'],middle_a.shape[0]))

#Synthetic Middle
middle_s = pd.read_csv("fairface_label_train_w_synth.csv")
middle_s = middle_s.drop(train_df.index[(train_df['age'] != '-1')],axis=0)
middle_s.drop(middle_s.index[(middle_s['race'] != 'Middle Eastern')],axis=0, inplace = True)
middle_s = middle_s.reset_index(drop = True)
middle_s = middle_a.drop(labels = range(all_races['S_Middle Eastern'],middle_a.shape[0]))

#Combine back into one
del train_df
combined = [white_a, black_a, latino_a, south_a, east_a, indian_a, middle_a, white_s, black_s, latino_s, south_s, east_s, indian_s, middle_s]
train_df = pd.concat(combined, ignore_index = True)
#train_df = train_df.reset_index(drop = True)
print(train_df)

print("trainset consists of ",train_df.shape)
print("test set consist of ",test_df.shape)

# Uncomment for race
train_df = train_df[['file', 'race']]
test_df = test_df[['file', 'race']]

# train_df = train_df[['file', 'gender']]
# test_df = test_df[['file', 'gender']]

train_df['file'] = 'FairFace/'+train_df['file']
test_df['file'] = 'FairFace/'+test_df['file']

train_df.head()
# Combine both Asian races into one for train and test
idx = train_df[(train_df['race'] == 'East Asian') | (train_df['race'] == 'Southeast Asian')].index
train_df.loc[idx, 'race'] = 'Asian'

idx = test_df[(test_df['race'] == 'East Asian') | (test_df['race'] == 'Southeast Asian')].index
test_df.loc[idx, 'race'] = 'Asian'

#Check distribution of each
100*train_df.groupby(['race']).count()[['file']]/train_df.groupby(['race']).count()[['file']].sum()

#convert from jpg to pixel values and store those
target_size = (224, 224)

def getImagePixels(file):
    #print(file)
    img = image.load_img(file, grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    return x

#make sure to Create a Folder called 'FairFace' and place the val, train, and imgs_jpg' folders inside

train_df['pixels'] = train_df['file'].progress_apply(getImagePixels)
test_df['pixels'] = test_df['file'].progress_apply(getImagePixels)

train_df.head()

#select test and train features from train_df and test_df

train_features = []; test_features = []

for i in range(0, train_df.shape[0]):
    train_features.append(train_df['pixels'].values[i])

for i in range(0, test_df.shape[0]):
    test_features.append(test_df['pixels'].values[i])

#convert to numpy
train_features = np.array(train_features)
train_features = train_features.reshape(train_features.shape[0], 224, 224, 3)

test_features = np.array(test_features)
test_features = test_features.reshape(test_features.shape[0], 224, 224, 3)

#convert to numpy
train_features = train_features / 255
test_features = test_features / 255

train_label = train_df[['race']]
test_label = test_df[['race']]

#find number of races present
races = train_df['race'].unique()

# change race name to numbers
for j in range(len(races)): #label encoding
    current_race = races[j]
    print("replacing ",current_race," to ", j+1)
    train_label['race'] = train_label['race'].replace(current_race, str(j+1))
    test_label['race'] = test_label['race'].replace(current_race, str(j+1))

train_label = train_label.astype({'race': 'int32'})
test_label = test_label.astype({'race': 'int32'})

#double check it worked
train_label.head()

#Set target for train and test
train_target = pd.get_dummies(train_label['race'], prefix='race')
test_target = pd.get_dummies(test_label['race'], prefix='race')

#double check
train_target.head()

#Split up training set into train and validation sets
train_x, val_x, train_y, val_y = train_test_split(train_features, train_target.values
                                        , test_size=0.15, random_state=18)

import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential

#Premade VGG-Face model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
 
model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

#pre-trained weights of vgg-face model. 
#you can find it here: https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
#related blog post: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/
model.load_weights('vgg_face_weights.h5')

# of races
num_of_classes = 6

# Just want to retrain top layers to save time and utilize transfer learning from better trained model
for layer in model.layers[:-7]:
    layer.trainable = False

base_model_output = Sequential()
base_model_output = Convolution2D(num_of_classes, (1, 1), name='predictions')(model.layers[-4].output)
base_model_output = Flatten()(base_model_output)
base_model_output = Activation('softmax')(base_model_output)

race_model = Model(inputs=model.input, outputs=base_model_output)

race_model.compile(loss='categorical_crossentropy'
                  , optimizer=keras.optimizers.Adam()
                  , metrics=['accuracy']
                 )

#setup model
checkpointer = ModelCheckpoint(
    filepath='race_model_single_batch.hdf5'
    , monitor = "val_loss"
    , verbose=1
    , save_best_only=True
    , mode = 'auto'
)

# How long to wait before exiting without improvement
patience = 10

val_scores = []; train_scores = []

# Train the classifier
enableBatch = True

epochs = 20 # duration

if enableBatch != True:
    early_stop = EarlyStopping(monitor='val_loss', patience=patience) 
    
    score = race_model.fit(
        train_x, train_y
        , epochs=epochs
        , validation_data=(val_x, val_y)
        , callbacks=[checkpointer, early_stop]
    )
    
else:
    batch_size = pow(2, 9)
    last_improvement = 0
    best_iteration = 0
    
    loss = 1000000 #initialize as a large value
    
    for i in range(0, epochs):
        
        print("Epoch ", i, ". ", end='')
        
        ix_train = np.random.choice(train_x.shape[0], size=batch_size)
        
        score = race_model.fit(
            train_x[ix_train], train_y[ix_train]
            , epochs=1
            , validation_data=(val_x, val_y)
            , callbacks=[checkpointer]
        )
        
        val_loss = score.history['val_loss'][0]
        train_loss = score.history['loss'][0]
        
        val_scores.append(val_loss)
        train_scores.append(train_loss)
        
        #--------------------------------
        
        if val_loss < loss:
            loss = val_loss * 1
            last_improvement = 0
            best_iteration = i * 1
        else:
            last_improvement = last_improvement + 1
            print("try to decrease val loss for ",patience - last_improvement," epochs more")
        
        #--------------------------------
        
        if last_improvement == patience:
            print("there is no loss decrease in validation for ",patience," epochs. early stopped")
            break

# Plot training and validation loss
if enableBatch != True:
    plt.plot(score.history['val_loss'][0:best_iteration], label='val_loss')
    plt.plot(score.history['loss'][0:best_iteration], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()
else:
    plt.plot(val_scores[0:best_iteration+1], label='val_loss')
    plt.plot(train_scores[0:best_iteration+1], label='train_loss')
    plt.legend(loc='upper right')
    plt.show()

# Load best model from training
from keras.models import load_model
race_model = load_model("race_model_single_batch.hdf5")

race_model.save_weights('race_model_single_batch.h5')

# test_perf = race_model.evaluate(test_features, test_target.values, verbose=1)
# print(test_perf)

# validation_perf = race_model.evaluate(val_x, val_y, verbose=1)
# print(validation_perf)

#Check model is robust
# abs(validation_perf[0] - test_perf[0]) < 0.01

# Run model on test data and save results
predictions = race_model.predict(test_features)

prediction_classes = []; actual_classes = []
right = []; wrong = []
gender_feature_test_Set = pd.read_csv("fairface_label_val.csv", nrows = 5000)['gender']
print(gender_feature_test_Set)

for i in range(0, predictions.shape[0]):
    prediction = np.argmax(predictions[i])
    prediction_classes.append(races[prediction])
    actual = np.argmax(test_target.values[i])
    actual_classes.append(races[actual])

#uncomment for visuals
    # if i in range(0,100):
    #     print(i)
    #     print("Actual: ",races[actual])
    #     print("Predicted: ",races[prediction])
        
    #     img = (test_df.iloc[i]['pixels'].reshape([224, 224, 3])) / 255
    #     plt.imshow(img)
    #     plt.show()
    #     print("----------------------")

prediction_classes_m = []; actual_classes_m = []
prediction_classes_f = []; actual_classes_f = []
right = []; wrong = []
test_df = pd.read_csv("fairface_label_val.csv", nrows = 5000)
gender_feature_test_Set = pd.read_csv("fairface_label_val.csv", nrows = 5000)['gender']
print(gender_feature_test_Set)
print(predictions.shape[0])

i=0

temp_m = [0]*len(races)
for i, r in enumerate(races):
  print(r)
  # get the actual number of values for each race
  df = test_df.loc[(test_df['race'] == r) & (test_df['gender']== 'Male')]
  temp_m[i] = df.shape[0]
print(temp_m)

temp_f = [0]*len(races)
for i, r in enumerate(races):
  print(r)
  # get the actual number of values for each race
  df = test_df.loc[(test_df['race'] == r) & (test_df['gender']== 'Female')]
  temp_m[i] = df.shape[0]
print(temp_f)

# Male 
for i in range(0, predictions.shape[0]):
    if gender_feature_test_Set[i] == 'Male':
      prediction = np.argmax(predictions[i])
      prediction_classes_m.append(races[prediction])
      actual = np.argmax(test_target.values[i])
      actual_classes_m.append(races[actual])

# Female
for i in range(0, predictions.shape[0]):
    if gender_feature_test_Set[i] == 'Female':
      prediction = np.argmax(predictions[i])
      prediction_classes_f.append(races[prediction])
      actual = np.argmax(test_target.values[i])
      actual_classes_f.append(races[actual])

#uncomment for visuals
    # if i in range(0,100):
    #     print(i)
    #     print("Actual: ",races[actual])
    #     print("Predicted: ",races[prediction])
        
    #     img = (test_df.iloc[i]['pixels'].reshape([224, 224, 3])) / 255
    #     plt.imshow(img)
    #     plt.show()
    #     print("----------------------")

#Histogram of predictions
from matplotlib import pyplot as plt
type(prediction_classes)
plt.hist(prediction_classes, 12)
plt.show()

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

#create confusion matrix to analyze model performance
cm0 = {'actual': actual_classes,'predicted': prediction_classes}
df = pd.DataFrame(cm0)
conf = pd.crosstab(df['actual'], df['predicted'], rownames=['Actual'], colnames=['Predicted'])
print(conf)

#create confusion matrix to analyze model performance
cm1 = {'actual_m': actual_classes_m,'predicted_m': prediction_classes_m}
df = pd.DataFrame(cm1)
conf1 = pd.crosstab(df['actual_m'], df['predicted_m'], rownames=['Actual_m'], colnames=['Predicted_m'])
print(conf1)

cm = confusion_matrix(actual_classes, prediction_classes)
df_cm = pd.DataFrame(cm, index=races, columns=races)

cm1 = confusion_matrix(actual_classes_m, prediction_classes_m)
df_cm1 = pd.DataFrame(cm1, index=races, columns=races)

#Visualize confusion matrix
confusion_matrix1 = confusion_matrix(actual_classes, prediction_classes, labels = races)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix1, display_labels = races)
plt.figure(figsize=(10,10))
disp.plot()
plt.show()

#Visualize confusion matrix
confusion_matrix_m = confusion_matrix(actual_classes_m, prediction_classes_m, labels = races)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_m, display_labels = races)
plt.figure(figsize=(10,10))
disp.plot()
plt.show()

#Visualize confusion matrix
confusion_matrix_f = confusion_matrix(actual_classes_f, prediction_classes_f, labels = races)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_f, display_labels = races)
plt.figure(figsize=(10,10))
disp.plot()
plt.show()