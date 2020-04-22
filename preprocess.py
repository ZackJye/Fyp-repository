import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import zipfile
import csv
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import scipy.io.wavfile as wav
from PIL import Image 
import torchvision.transforms as transforms
#Keras
import keras
import subprocess
import cv2
from PIL import Image 
from model import *

def getWav(filename,rootdir):
  command = "ffmpeg -i {}/{} -ab 320k -ac 2 -ar 44100 -vn {}.wav".format(rootdir,filename,filename)
  subprocess.call(command, shell=True)   
  
def getAudioFeature(filename,rootdir):
        features=[]
        songname = f'{rootdir}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=7)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features.append(np.mean(chroma_stft))
        features.append(np.mean(rmse))
        features.append(np.mean(spec_cent))
        features.append(np.mean(spec_bw))
        features.append(np.mean(rolloff))
        features.append(np.mean(zcr))
        for e in mfcc:
            features.append(np.mean(e))
        audio = np.array(features)
        audio = audio.astype('float').reshape(-1, 26)
        return audio
def processImage(filename,rootdir,destdir):
  cap = cv2.VideoCapture(rootdir+'/'+filename);
  fps = cap.get(cv2.CAP_PROP_FPS)
  l_test=(int(fps)*7)  # get seven second of video
  l4=np.linspace(1,l_test,15).astype(int) 
  file_name=(filename.split('.mp4'))[0]
  ## Setting the frame limit to 210
  cap.set(cv2.CAP_PROP_FRAME_COUNT, 210)
  length=430
  count=0
  name_count=1
  ## Running a loop to each frame and saving it in the created folder
  while(cap.isOpened()):
      count+=1
      if length==count:
          break
      ret, frame = cap.read()
      if frame is None:
          continue

      ## Resizing it to 224*224 to save the disk space and fit into the model
      if(count in (l4)):
          frame = cv2.resize(frame,(224, 224))
      # Saves image of the current frame in jpg file
          name = str(destdir)+str(file_name)+'_' + str(name_count) + '.jpg'
          name_count+=1
          cv2.imwrite(name, frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
def getResult(image_path,image_name,audio,model):
        X=[]
        for i in range(1,16):

          img_name = os.path.join(image_path, image_name)
          image = Image.open(str(img_name)+'_'+str(i)+'.jpg')
          transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
          
          image = transform(image)
          X.append(image)
        X = torch.stack(X, dim=0)
        X=X.unsqueeze(0)
        audio = torch.from_numpy(audio)
        audio = audio.type(torch.cuda.FloatTensor)
        audio = audio.to(device)
        audio=audio.unsqueeze(0)
        model.eval()
        y=model(x1=X,x2=audio)
        return y

