import predict
from flask import Flask, flash, jsonify, request, render_template, redirect, url_for
from PIL import Image
import numpy
import cv2
import werkzeug
import os
from utils import data_uri_to_cv2_img

input_img_path = 'static/input_img.png'
app = Flask(__name__)

@app.route("/")
def firstpage():
    return render_template('canvas.html')

@app.route("/doodlepredict",methods=["GET", "POST"])
def predictedclasses():
    imgstring = request.form.get('data')
    img = data_uri_to_cv2_img(imgstring)
    cv2.imwrite(input_img_path, img)
    objs, pred_accuracy = predict.predict_doodle(input_img_path)
    return str(objs+pred_accuracy)

if __name__ == "__main__":
    app.run(host= '0.0.0.0', port=5000, debug=False)
