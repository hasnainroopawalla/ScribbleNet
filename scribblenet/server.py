import argparse
import base64
import os
import time
import uuid

import cv2
import numpy as np
import tensorflow as tf  # tensorflow == 1.15.0
from flask import (Flask, redirect, render_template, request, json,
                   send_from_directory, url_for)
from keras.models import Sequential, load_model
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from PIL import Image, ImageOps
from tensorflow import keras  # keras == 2.2.5
from io import BytesIO
import io
from typing import List, Dict, Union, ByteString, Any

input_img_path = "static/input_img.png"
app = Flask(__name__)


@app.route("/")
def firstpage():
    return render_template("canvas.html")

def initialize():
    #model_path = '../models/'
    model = tf.keras.models.load_model('/Users/hasnain/Projects/ScribbleNet/scribblenet/models/e25.h5') #100_classes_16000_per_class.h5

    class_names = ['Smiley Face', 'Alarm Clock', 'Donut', 'Butterfly', 'Hat', 'Wristwatch', 'Paper Clip', 'Rainbow', 'Microphone', 'Screwdriver',
                   'Chair', 'Spoon', 'Snake', 'Cloud', 'Scissors', 'Eye', 'Bridge', 'Suitcase', 'Light Bulb', 'Mushroom', 'Book', 'Dumbbell',
                   'Flower', 'Laptop', 'Ice Cream', 'Lightning', 'Sword', 'Tree', 'Headphones', 'Moon', 'Baseball Bat', 'Ceiling Fan', 'Sun',
                   'Bed', 'Cup', 'Table', 'Hammer', 'Pants', 'Lollipop', 'Ladder', 'Tennis Racquet', 'Cat', 'Car', 'Fan', 'T-shirt', 'Umbrella',
                   'Bench', 'Airplane', 'Envelope', 'Coffee Cup', 'Hot Dog', 'Pizza', 'Cell Phone', 'Radio', 'Baseball', 'Camera', 'Bird',
                   'Star', 'Spider', 'Pencil', 'Key', 'Mountain', 'Circle', 'Cookie', 'Candle', 'Sock', 'Triangle', 'Basketball', 'Knife',
                   'Apple', 'Clock']

    # class_names = ['smiley_face', 'alarm_clock', 'donut', 'butterfly', 'hat', 'wristwatch', 'paper_clip', 'rainbow', 'microphone', 'screwdriver',
    #                'chair', 'spoon', 'snake', 'cloud', 'scissors', 'eye', 'bridge', 'suitcase', 'light_bulb', 'mushroom', 'book', 'dumbbell',
    #                'flower', 'laptop', 'ice_cream', 'lightning', 'sword', 'tree', 'headphones', 'moon', 'baseball_bat', 'ceiling_fan', 'sun',
    #                'bed', 'cup', 'table', 'hammer', 'pants', 'lollipop', 'ladder', 'tennis_racquet', 'cat', 'car', 'fan', 't-shirt', 'umbrella',
    #                'bench', 'airplane', 'envelope', 'coffee_cup', 'hot_dog', 'pizza', 'cell_phone', 'radio', 'baseball', 'camera', 'bird',
    #                'star', 'spider', 'pencil', 'key', 'mountain', 'circle', 'cookie', 'candle', 'sock', 'triangle', 'basketball', 'knife',
    #                'apple', 'clock']


##    class_names = ['Saw', 'Coffee Cup', 'Power Outlet', 'Microphone', 'Triangle', 'Cat', 'Paper Clip',
##                'Drums', 'Diving Board', 'Sun', 'Scissors', 'Butterfly', 'Ladder', 'Beard', 'Helmet',
##                'Bicycle', 'Face', 'Eye', 'Syringe', 'Bed', 'Smiley Face', 'Sword', 'Door', 'Spider',
##                'Bridge', 'Spoon', 'Fan', 'Cell Phone', 'Donut', 'Rifle', 'Baseball Bat', 'Camera', 
##                'Baseball', 'Screwdriver', 'Table', 'Anvil', 'Frying_pan', 'Tree', 'Chair', 'Tooth',
##                'Rainbow', 'Bench', 'Star', 'Ceiling Fan', 'Headphones', 'Moon', 'Key', 'Eyeglasses',
##                'Lollipop', 'Cookie', 'Hat', 'Shorts', 'Grapes', 'Pencil', 'Hot Dog', 'Bird', 'Basketball',
##                'Hammer', 'Radio', 'T-shirt', 'Pizza', 'Shovel', 'Flower', 'Clock', 'Wristwatch', 'Tent',
##                'Ice Cream', 'Airplane', 'Mushroom', 'Wheel', 'Bread', 'Mountain', 'Axe', 'Stop Sign', 
##                'Lightning', 'Car', 'Laptop', 'Snake', 'Dumbbell', 'Sock', 'Cup', 'Moustache', 'Book', 
##                'Traffic Light', 'Umbrella', 'Line', 'Suitcase', 'Circle', 'Candle', 'Pants', 'Tennis_racquet',
##                'Alarm Clock', 'Square', 'Pillow', 'Cloud', 'Broom', 'Knife', 'Light Bulb', 'Apple', 'Envelope']
    print('LOADED')
    return class_names, model

def predict_doodle(img):
    class_names, model = initialize()
    size = 28, 28
    # img = load_img(img_path, target_size=(28,28))
   
    img = img.resize(size)

    img = ImageOps.invert(img.convert('RGB'))
    img_tensor = img_to_array(img)
    
    img_tensor = cv2.cvtColor(img_tensor, cv2.COLOR_BGR2GRAY) # Convert channel to 1 (grayscale)
    
    img_tensor = np.expand_dims(img_tensor, axis=2) # Add last channel as 1  (28,28) to (28,28,1)
    
    img_tensor = np.expand_dims(img_tensor, axis=0) # Add 1 more channel at start to specify number of input images (1,28,28,1)
    
    img_tensor /= 255. 
    

    pred = model.predict(img_tensor)[0]
    ind = (-pred).argsort()[:5]  
    acc = sorted(pred, reverse = True)[:5]  # Accuracy of the top 5 predictions
    objs = [class_names[x] for x in ind]
    pred_accuracy = []
    for i in acc:
        pred_accuracy.append(round((i*100),2))
    model = None
    return objs,pred_accuracy

@app.route("/predict", methods=["POST"])
def predict():
    img_str = request.form.get("data")
    print(img_str,type(img_str))
    
    if img_str == '1':
        print('Connection Initialized')
        return json.dumps({"status":"Connected"})

    print('Received')
    img = base64.b64decode(img_str.split(',')[1])
    image = Image.open(BytesIO(img))
    # print(type(image))
    start = time.time()
    result = predict_doodle(image)
    end = time.time()

    f = {}
  
    for i in range(len(result[0])):
        f[str(i)] = result[0][i]
    print(f,result[1],result[1][0])
    f['time'] = 'Time Taken: '+str(round(end-start,2))+' seconds'
    f['conf'] = result[1][0]
    image = None
    return json.dumps(f)


# @app.route("/doodlepredict", methods=["GET", "POST"])
# def predictedclasses():

#     imgstring = request.form.get("data")
#     img = data_uri_to_cv2_img(imgstring)
#     cv2.imwrite(input_img_path, img)
#     objs, pred_accuracy = predict.predict_doodle(input_img_path)
#     return str(objs + pred_accuracy)

def main():
    app.run(host="0.0.0.0", port=9000, debug=False)
