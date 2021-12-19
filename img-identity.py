# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:24:38 2021

@author: Admin
"""

#pip install -q gradio
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import gradio as gr
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
#pip install tensorflow
#pip install tensorflow_hub
TF_MODEL_URL = 'https://tfhub.dev/google/on_device_vision/classifier/landmarks_classifier_asia_V1/1'
LABEL_MAP_URL = 'https://www.gstatic.com/aihub/tfhub/labelmaps/landmarks_classifier_asia_V1_label_map.csv'
IMAGE_SHAPE = (321, 321)
classifier = tf.keras.Sequential([hub.KerasLayer(TF_MODEL_URL,
                                                 input_shape=IMAGE_SHAPE+(3,),
                                                 output_key="predictions:logits")])

df = pd.read_csv(LABEL_MAP_URL)
label_map = dict(zip(df.id, df.name))
img_loc = "Image.jpeg"
img = Image.open("C:\\Users\\Admin\\Downloads\\Image.jpg").resize(IMAGE_SHAPE)
img 
