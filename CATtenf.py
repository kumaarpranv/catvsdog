import cv2
import numpy as np
import os         
from random import shuffle  
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tqdm import tqdm
from pathlib import Path
import models

TRAINDIR='train/train'
TESTDIR='test1/test1'
PRACTDIR='pract'
IMG_SIZE= 50
LR=1e-3


def predict_clf(target_path,model):
    target_img=cv2.imread(target_path)
    target= process_img(target_path)
    np_test=np.array(target).reshape(IMG_SIZE, IMG_SIZE, 1)
    ans=model.predict([np_test])
    sc=ans
    ans=np.round(ans)
    print(sc[0])
    if(np.all(ans==[1,0])):
        return 'cat',str(sc[0]*100)[1:6]
    else:
        return 'dog',str(sc[0]*100)[1:6]

def process_img(path):
    img_data=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE))
    return np.array(img_data)

def create_label(img_name):
    #creating one-hot encoded vector from imgname
    word_label=img_name.split('.')[0]
    if word_label == 'cat':
        return np.array([1,0])
    elif word_label == 'dog':
        return np.array([0,1])

def model_evaluate(model,test_it_x,test_it_y):
    test_it_y=np.array(test_it_y)
    score = model.evaluate(test_it_x, test_it_y)
    return score
 


def run(targname=''):
   

    MODEL_NAME='default.tfl'
    MODEL_DIR='models'

    #tf-model
    tf.reset_default_graph()
    model=models.catDefaultModel(IMG_SIZE,LR)

    
    if(os.path.exists( os.path.join( MODEL_DIR,'{}.meta'.format(MODEL_NAME)))):
        model.load(os.path.join( MODEL_DIR,MODEL_NAME))
    
    if not targname=='':
        target_path=targname
        return predict_clf(target_path,model)
        #print(model_evaluate(model,X_test,y_test))

run()