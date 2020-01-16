import os
import glob
import numpy as np
from tensorflow import keras 
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randint
from PIL import Image, ImageOps
import cv2
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

def initialize():
    model_path = '../models/'
    model = tf.keras.models.load_model(model_path+'100_classes_16000_per_class.h5')

    class_names = ['saw', 'coffee_cup', 'power_outlet', 'microphone', 'triangle', 'cat', 'paper_clip',
                'drums', 'diving_board', 'sun', 'scissors', 'butterfly', 'ladder', 'beard', 'helmet',
                'bicycle', 'face', 'eye', 'syringe', 'bed', 'smiley_face', 'sword', 'door', 'spider',
                'bridge', 'spoon', 'fan', 'cell_phone', 'donut', 'rifle', 'baseball_bat', 'camera', 
                'baseball', 'screwdriver', 'table', 'anvil', 'frying_pan', 'tree', 'chair', 'tooth',
                'rainbow', 'bench', 'star', 'ceiling_fan', 'headphones', 'moon', 'key', 'eyeglasses',
                'lollipop', 'cookie', 'hat', 'shorts', 'grapes', 'pencil', 'hot_dog', 'bird', 'basketball',
                'hammer', 'radio', 't-shirt', 'pizza', 'shovel', 'flower', 'clock', 'wristwatch', 'tent',
                'ice_cream', 'airplane', 'mushroom', 'wheel', 'bread', 'mountain', 'axe', 'stop_sign', 
                'lightning', 'car', 'laptop', 'snake', 'dumbbell', 'sock', 'cup', 'moustache', 'book', 
                'traffic_light', 'umbrella', 'line', 'suitcase', 'circle', 'candle', 'pants', 'tennis_racquet',
                'alarm_clock', 'square', 'pillow', 'cloud', 'broom', 'knife', 'light_bulb', 'apple', 'envelope']
    return class_names, model

def predict_doodle(img_path):
    class_names, model = initialize()
    img = load_img(img_path, target_size=(28,28))
    
    img = ImageOps.invert(img)
    #plt.imshow(img) 
    #plt.show()
    img_tensor = img_to_array(img)

    
    img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2GRAY) # Convert channel to 1 (grayscale)
    img_tensor = np.expand_dims(img_tensor, axis=2) # Add last channel as 1  (28,28) to (28,28,1)
    img_tensor = np.expand_dims(img_tensor, axis=0) # Add 1 more channel at start to specify number of input images (1,28,28,1)
    img_tensor /= 255. 

    #print(img_tensor.shape)

    pred = model.predict(img_tensor)[0]
    ind = (-pred).argsort()[:5]  
    acc = sorted(pred, reverse = True)[:5]  # Accuracy of the top 5 predictions
    objs = [class_names[x] for x in ind]
    pred_accuracy = []
    for i in acc:
        pred_accuracy.append(round((i*100),2))

    return objs,pred_accuracy
