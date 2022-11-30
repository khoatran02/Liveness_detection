
import seaborn as sns
import os
import pandas as pd
from IPython.display import Image
import json
import numpy as np
import natsort
from google.colab.patches import cv2_imshow
import cv2
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from mtcnn import MTCNN


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
      {'paths' : paths,
       'fname' : Videos_name
      })
  
  
  # df = df.sort_values(by=['fname'])
  # df = df.reset_index()
  # df = df.drop(columns=["index"])
  return df

df_paths = get_video_Path("/content/raw_data/train/videos")
df_label = pd.read_csv("/content/raw_data/train/label.csv")

df_paths_label = pd.merge(df_paths, df_label, on = ['fname'] )


df_paths_fake = df_paths_label[df_paths_label["liveness_score"] == 0]
df_paths_fake.reset_index(inplace=True, drop = True)

df_paths_real = df_paths_label[df_paths_label["liveness_score"] == 1]
df_paths_real.reset_index(inplace= True, drop = True)


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

def splitFrame(dataframe, path_folder_save):
  df = dataframe.copy()

  for i in range(len(df)):
    print("step: ",i)
    listframe=[]
    cam = cv2.VideoCapture(df.loc[i,"paths"])
    
    # slip file name
    s = df.loc[i,"paths"]
    s = s.split("/")[-1]
    s = s.split(".mp4")[0]

 
    while(True):
      # reading from frame
      ret,frame = cam.read()
      if ret:
        listframe.append(frame)
      else:
        break

    first_pos = 0
    middle_pos = int((len(listframe)-1)/2)
    last_pos = len(listframe)-1

    detector = MTCNN()

    face_number = 0 

    while(True):        
      img = cv2.cvtColor(listframe[first_pos], cv2.COLOR_BGR2RGB)
      detections = detector.detect_faces(img)

    # print(detections)
    # plt.figure(figsize = (10,10))
    # plt.imshow(img)
    # plt.axis('off')

      if detections !=[] :
        image = crop_image(listframe[first_pos], detections)
        first_pos_name = path_folder_save +"/" + str(first_pos) + "_" + s + '.jpg'
        cv2.imwrite(first_pos_name, image)
        face_number = face_number + 1
        break
      else:
        first_pos = first_pos + 1
      if first_pos == (len(listframe)-1):
        break

    if face_number != 0:
      while(True):
        img = cv2.cvtColor(listframe[middle_pos], cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(img)
        if detections !=[]:
          image = crop_image(listframe[middle_pos], detections)
          middle_pos_name =  path_folder_save +"/" + str(middle_pos) + "_" + s + '.jpg'
          cv2.imwrite(middle_pos_name, image)
          middle_pos = middle_pos + 1
          break
        else:
          middle_pos = middle_pos + 1

        if middle_pos == (len(listframe)-1):
          break

      while(True):
        img = cv2.cvtColor(listframe[last_pos], cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(img)
        if detections !=[]:
          image = crop_image(listframe[last_pos], detections)
          last_pos_name =  path_folder_save +"/" + str(last_pos) + "_" + s + '.jpg'
          cv2.imwrite(last_pos_name, image)
          break
        else:
          last_pos = last_pos - 1
        if last_pos == middle_pos:
          break

splitFrame(df_paths_real, "/content/full_face/real")
splitFrame(df_paths_fake, "/content/full_face/fale")

"""#Crop face test"""

df_test = get_video_Path("/content/raw_data/public")



def create_df_image_Path(video_path):
  paths = []
  image_name = []
  for dirname, _, filenames in os.walk(video_path):
      for filename in filenames:
          paths.append(os.path.join(dirname, filename))
          image_name.append(filename)
          
  paths = natsort.natsorted(paths,reverse=False)
  image_name = natsort.natsorted(image_name,reverse=False)
  
  df = pd.DataFrame(
      {'paths' : paths,
       'fname' : image_name
      })
  
  return df

df_real_face = create_df_image_Path("/content/full_face_data/real")
df_fake_face = create_df_image_Path("/content/full_face_data/fake")

df_real_face.loc[0,"fname"]


# slip file name
s = df_real_face.loc[0,"paths"]
s = s.split("/")[-1]
s = s.split(".mp4")[0]

def cropMask(df, path_save):
  for i in range(len(df)):
    img = cv2.imread(df.loc[i,"paths"])
    y=0
    x=0
    h, w, c = img.shape
    h = int(h/1.75)
    # h = 750
    Image_crop = img[y:y+h, x:x+w]
                        
    path_name = path_save +"/" + df.loc[i,"fname"] 

    cv2.imwrite(path_name, Image_crop)

cropMask(df_real_face, "/content/face_without_mask/real")
cropMask(df_fake_face, "/content/face_without_mask/fake")

img = cv2.imread("/content/full_face_data/real/0_1031.jpg")

y=0
x=0
h, w, c = img.shape
h = int(h/1.75)
# h = 750
Image_crop = img[y:y+h, x:x+w]
cv2_imshow(Image_crop)

img = cv2.imread("/content/face_without_mask/real/0_1031.jpg")

y=0
x=200
h, w, c = img.shape
x = int(w/2)
# h = 750
Image_crop = img[y:y+h, x:w]
cv2_imshow(Image_crop)

h, w, c = img.shape


