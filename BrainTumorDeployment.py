# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 09:40:37 2021

@author: Pavan K M
"""

import streamlit as st
import os
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
bt=tf.keras.models.load_model('Brain_tumor_pred.h5')

def names(number):
    if number==0:
        return str('a Tumor')
    else:
        return str('not a tumor')
    
def optn(choice):
    p_cnt=0
    p_id='BT'
    df=pd.DataFrame(columns=['pname','age','path','pid'])
    if(choice=='Scan Brain Image for Tumor'):
        p_cnt+=1
        pname=st.text_input(label='Enter Patient Name')
        age=st.text_input(label='Enter age of the Patient')
        data=st.file_uploader('Upload Image in JPG',type=['jpg'])
        Scan=st.button('Start Scan')
        if(Scan):
            if(data is not None ):
                img=Image.open(data)
                x = np.array(img.resize((128,128)))
                x= x.reshape(1,128,128,3)
                res = bt.predict_on_batch(x)
                classification = np.where(res == np.amax(res))[1][0]
                #st.markdown(imshow(img))
                st.image(img)
                conf=str(res[0][classification]*100)+ '%Confidence This Is '+names(classification)
                st.title(conf)
                patient_id=p_id+str(p_cnt)
                path='C:/Users/Pavan K M/Patient BT image'+p_id+str(p_cnt)+".jpg"
                img.save(path)
                pdata={'pname':pname,'age':age,'path':path,'pid':patient_id}
                df=df.append(pdata,ignore_index=True)
                st.write(df)
                choice=st.selectbox('Brain Scanning or Fetch Report', ['Select','Scan Brain Image for Tumor','Fetch Details'],key=p_cnt)
                if(choice=='Scan Brain Image for Tumor'):
                    optn(choice)
                elif(choice=='Fetch Details'):
                    st.write('coming soon')
df=pd.DataFrame(columns=['pname','age','path','pid'])
choice=st.selectbox('Brain Scanning or Fetch Report', ['Select','Scan Brain Image for Tumor','Fetch Details'])   
if(choice=='Scan Brain Image for Tumor'):
    optn(choice)
  
'''p_cnt=0
p_id='BT'
if(choice=='Scan Brain Image for Tumor'):
    p_cnt+=1
    pname=st.text_input(label='Enter Patient Name')
    age=st.text_input(label='Enter age of the Patient')
    data=st.file_uploader('Upload Image in JPG',type=['jpg'])
    Scan=st.button('Start Scan')
    if(Scan):
        if(data is not None ):
            img=Image.open(data)
            x = np.array(img.resize((128,128)))
            x= x.reshape(1,128,128,3)
            res = bt.predict_on_batch(x)
            classification = np.where(res == np.amax(res))[1][0]
            #st.markdown(imshow(img))
            st.image(img)
            conf=str(res[0][classification]*100)+ '%Confidence This Is '+names(classification)
            st.title(conf)
            patient_id=p_id+str(p_cnt)
            path='C:/Users/Pavan K M/Patient BT image'+p_id+str(p_cnt)+".jpg"
            img_save=img.save(path)
            pdata={'pname':pname,'age':age,'path':path,'pid':patient_id}
            df=df.append(pdata,ignore_index=True)
            st.write(df)
            choice=st.selectbox('Brain Scanning or Fetch Report', ['Select','Scan Brain Image for Tumor','Fetch Details'])
            '''
'''
data=st.file_uploader('Upload Image in JPG',type=['jpg'])


if(data is not None ):
    img=Image.open(data)
    x = np.array(img.resize((128,128)))
    x= x.reshape(1,128,128,3)
    res = bt.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    #st.markdown(imshow(img))
    st.image(img)
    conf=str(res[0][classification]*100)+ '%Confidence This Is '+names(classification)
    st.title(conf)
    #st.markdown(names(classification)'''