# from google.colab import drive
# drive.mount('/content/drive')

"""#Load test data"""

# !mkdir test_data
# !unzip -q "/content/drive/MyDrive/Colab Notebooks/Competition/Zalo_AI_Challenge_2022/Data/public_test.zip" -d "/content/test_data"

"""#Import Library """

# !pip install mtcnn

import seaborn as sns
import os
import pandas as pd
from IPython.display import Image
import json
import numpy as np
import natsort
# from google.colab.patches import cv2_imshow
import cv2
import warnings
import time
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from tensorflow.keras.models import model_from_json

"""#Load model"""

json_file = open("save_models/Model_with_mask_01.json",'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights("save_models/Model_weights_with_mask_01.h5")
print("Model loaded from disk")

"""#Perdict"""

def get_video_Path(video_path):
  paths = []
  Videos_name = []
  for dirname, _, filenames in os.walk(video_path):
      for filename in filenames:
          paths.append(os.path.join(dirname, filename))
          Videos_name.append(filename)
          
  paths = natsort.natsorted(paths,reverse=False)
  Videos_name = natsort.natsorted(Videos_name,reverse=False)
  
  df = pd.DataFrame(
      {
        'paths' : paths,
        'fname' : Videos_name
      })

  return df

df_paths = get_video_Path("./data/")

#Prediction Function
def predict(file):
  resized_face = cv2.resize(file,(224,224))
  resized_face = resized_face.astype("float") / 255.0
  # resized_face = img_to_array(resized_face)
  resized_face = np.expand_dims(resized_face, axis=0)
  # pass the face ROI through the trained liveness detector
  # model to determine if the face is "real" or "fake"
  array = model.predict(resized_face)
  result = array[0]
  # answer = np.argmax(result)
  # if answer == 1:
  #   print("Predicted: 1")
  # elif answer == 0:
  #   print("Predicted: 0")

  return result[1]

def crop_image(image,data):
    print(data)
    biggest=0
    for faces in data:
        box=faces['box']            
        # calculate the area in the image
        area = box[3]  * box[2]
        if area>biggest:
            biggest=area
            bbox=box 
    bbox[0]= 0 if bbox[0]<0 else bbox[0]
    bbox[1]= 0 if bbox[1]<0 else bbox[1]
    image=image[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]] 
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert from bgr to rgb
    return image

def splitFrame(dataframes):

  paths = []
  fname = []
  liveness_score = []
  predicted_times = []

  detector = MTCNN()

  for i in range(len(dataframes)):

    paths.append(dataframes.loc[i,"paths"])
    fname.append(dataframes.loc[i,"fname"])

    print("step: ",i)
    listframe=[]
    cam = cv2.VideoCapture(dataframes.loc[i,"paths"])
 
    while(True):
      # reading from frame
      ret,frame = cam.read()
      if ret:
        listframe.append(frame)
      else:
        break

    face_number = 0

    for i in range(len(listframe)):
      img = cv2.cvtColor(listframe[i], cv2.COLOR_BGR2RGB)
      detections = detector.detect_faces(img)
      t1 = time.time()
      if detections !=[]:
        t1 = time.time()
        image = crop_image(listframe[i], detections)
        liveness_score.append(predict(image))
        print(predict(image))
        face_number = face_number + 1
        break
    
    if face_number == 0:
      t1 = time.time()
      liveness_score.append(0)
    
    t2 = time.time()
    predicted_time = int(t2*1000 - t1*1000)
    predicted_times.append(predicted_time)
  
  dict_0 = {"paths" : paths, "fname" : fname, "liveness_score" : liveness_score, "predicted_times" : predicted_times}
  df = pd.DataFrame(dict_0)
  # df = df.sort_values(by=['paths'], ascending=True)
  df = df.reset_index(drop=True) 

  return df

df_result = splitFrame(df_paths)

df_predict = df_result.drop(columns=["paths", "predicted_times"])
# df_predicted_time = df_result.drop(columns=["paths", "liveness_score"])


import os
directory = "result"
if not os.path.exists(directory):
    os.makedirs(directory)

df_predict.to_csv("./result/submission.csv",header=True, index=False)
# df_predicted_time.to_csv("./result/time_submission.csv",header=True, index=False)


























