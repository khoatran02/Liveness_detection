
import seaborn as sns
import os
import pandas as pd
import json
import numpy as np
import natsort 
import zipfile
import cv2
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow
# from mtcnn import MTCNN
import cv2


def create_df_video_Path(video_path):
  paths = []
  Videos_name = []
  for dirname, _, filenames in os.walk(video_path):
      for filename in filenames:
          paths.append(os.path.join(dirname, filename))
          Videos_name.append(filename)
          
  paths = natsort.natsorted(paths,reverse=False)
  Videos_name = natsort.natsorted(Videos_name,reverse=False)
  
  df = pd.DataFrame(
      {'paths' : paths,
       'fname' : Videos_name
      })
  
  return df

df_paths_real_face = create_df_video_Path("/content/data_face_with_mask/real")
df_paths_fake_face = create_df_video_Path("/content/data_face_with_mask/fake") 

df_paths_real_face.loc[:,"label"] = 1

df_paths_fake_face.loc[:,"label"] = 0

df_data = pd.concat([df_paths_fake_face,df_paths_real_face], ignore_index=True)

img_size = 224 ## ImageNet ==> 224 x 224
def extract_features(df):
  arr_features = []
  arr_label = []
  for i in range(len(df)): 
    img_array = cv2.imread(df.loc[i,"paths"])
    new_array = cv2.resize(img_array, (img_size, img_size))
    label = df.loc[i,"label"]
    arr_features.append(new_array)
    arr_label.append(label)
  return arr_features, arr_label

arr_features, arr_label = extract_features(df_data)

# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data_features = np.array(arr_features, dtype="float") / 255.0

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
labels = le.fit_transform(arr_label)
labels = np_utils.to_categorical(labels, 2)

len(labels)

from sklearn.model_selection import train_test_split

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(x_train, x_test, Y_train, Y_test) = train_test_split(data_features, labels,
	test_size=0.3, random_state=42)

from keras.preprocessing.image import ImageDataGenerator
# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")



"""#Model Training"""

import tensorflow
from keras.layers import Dense,Dropout,Input,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large
from keras.models import model_from_json
import json

"""##MobileNetV2"""

mobilenet = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))

mobilenet.trainable = False

output = Flatten()(mobilenet.output)
output = Dropout(0.3)(output)
output = Dense(units = 8,activation='relu')(output)
prediction = Dense(1,activation='sigmoid')(output)

model = Model(inputs = mobilenet.input,outputs = prediction)
model.summary()

# tell the model what cost and optimization method to use
model.compile(
  loss='binary_crossentropy',
  optimizer=Adam(
    learning_rate=0.000001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07
),
  metrics=['accuracy']
)

history = model.fit(x_train, Y_train, validation_data=(x_test,Y_test), batch_size=25, epochs = 450) ## Training

epochs = list(range(450))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, label='Training Accuracy')
plt.plot(epochs, val_acc, label='Validation Accuracy')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

epochs = list(range(450))
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# serialize model to JSON
model_json = model.to_json()

with open("save_models/Model_without_mask_01.json", "w") as json_file:
    json_file.write(model_json)

# model.save('/content/drive/MyDrive/Colab Notebooks/Competition/Zalo_AI_Challenge_2022/Model/Model_with_mask_01.h5')
model.save_weights('save_models/Model_weights_with_mask_01.h5')






































