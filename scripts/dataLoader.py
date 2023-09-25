import os
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf
import cv2
from sklearn.utils import shuffle


def get_face_data(num_people, face_angles, path='Data'):
  face = []
  y = []
  for i in tqdm(range(num_people)):
    if i<9:
      face_folder = path+'/CasiaBFaces/'+ 'person00'+ str(i+1) + '/'
    elif i>=9 and i<99:
      face_folder = path+'/CasiaBFaces/'+ 'person0'+ str(i+1) + '/'
    elif i>=100:
      face_folder = path+'/CasiaBFaces/'+ 'person'+ str(i+1) + '/'

    for angle in face_angles:
      angle_folder = face_folder + angle + '/'
      images = os.listdir(angle_folder)
      images.sort()
      images = images[:24]
      # images = random.sample(images, 2)
      # images = images[-1:]

      for image in images:
            path = angle_folder + image
            # img = cv2.imread(path,-1)
            img = cv2.imread(path, 0)
            img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
            img = np.array(img, dtype=np.float16)
            img = np.stack((img,)*3, axis=-1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (32, 32))
            face.append(img)
            y.append(i)

  face = np.array(face, dtype = np.float32)
  y = np.array(y, dtype = np.float16)
  y= tf.keras.utils.to_categorical(y)

  face,y = shuffle(face,y)
  return face,y

def get_gait_data(num_people, angles, train=False):
    x1 = []
    y = []
    sub_fold_list = sorted(os.listdir(f'CasiaB'))[:num_people]
    if train:
        m_angles, sub_angles = angles[:3], angles[3:]
        x2, y2 = [], []
    else:
        m_angles, sub_angles = angles, 0
    for j in tqdm(sub_fold_list, total=len(sub_fold_list)):
        x = []
        for k in m_angles:
            img_fold = sorted(os.listdir(f'CasiaB/{j}/{k}'))[:24]
            for impth in img_fold:
                # print(f'CasiaB/{j}/{k}/{impth}')
                img = cv2.imread(f'CasiaB/{j}/{k}/{impth}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
                img = np.array(img, dtype=np.float16)
                x.append(img)
        gait = np.array(x)
        x1.append(gait)
        y.append(int(j.split('person')[1])-1)
        
        if train:
            if len(sub_angles)>3 and len(sub_angles)%3==0:
                n_angles_loop = len(sub_angles)
            else:
                n_angles_loop = 1
            for loop in range(0,n_angles_loop, 3):
                x_1=[]
                assert sub_angles==0 or len(sub_angles)>0, "There is no subject to train extra data on. Set train=False for this case"
                for k in sub_angles[loop:loop+3]:
                    img_fold = sorted(os.listdir(f'CasiaB/{j}/{k}'))[:24]
                    for impth in img_fold:
                        # print(f'CasiaB/{j}/{k}/{impth}')
                        img = cv2.imread(f'CasiaB/{j}/{k}/{impth}')
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
                        img = np.array(img, dtype=np.float16)
                        x_1.append(img)
                gait = np.array(x_1)
                x2.append(gait)
                y2.append(int(j.split('person')[1])-1)
    if train:
        x1.extend(x2)
        y.extend(y2)
    x1 = np.array(x1, dtype = np.float16)
    y = tf.keras.utils.to_categorical(y)
    x1, y = shuffle(x1, y)
    
    return x1,y