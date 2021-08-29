# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import keras
import cv2
from keras.applications.vgg16 import VGG16
import numpy as np
import pickle as pk
import urllib.request
from sklearn import preprocessing
from urllib.error import HTTPError
import tensorflow as tf


app = Flask(__name__)

SIZE = 256
lable_encode = ['goku', 'luffy', 'naruto', 'natsu']
VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))
pca_reload = pk.load(open("pca.pkl",'rb'))
model = tf.keras.models.load_model('saved_model')

def predict_image(url,VGG_model,pca_reload):
    SIZE = 256  #Resize images
    # Capture train data and labels into respective lists
    one_image = []
    pred_image = urllib.request.urlretrieve(url,"local-img.jpg")

    img = cv2.imread(pred_image[0], cv2.IMREAD_COLOR)       
    img = cv2.resize(img, (SIZE, SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    one_image.append(img)

    #Convert lists to arrays        
    one_image = np.array(one_image)

    one_image =  one_image / 255.0
    VGG_model = VGG_model

    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG_model.layers:
        layer.trainable = False

    new_feature_extractor=VGG_model.predict(one_image)
    new_features = new_feature_extractor.reshape(new_feature_extractor.shape[0], -1)
    #from sklearn.decomposition import PCA
    # later reload the pickle file
    pca_reload = pca_reload

    #result_new = pca_reload .transform(X)
    image_PCA = pca_reload.transform(new_features) #Make sure you are just transforming, not fitting. 
    res = model.predict(image_PCA)
    predict_res = np.argmax(res, axis=1)

    
    return predict_res[0]


images = ["https://cdn-www.comingsoon.net/assets/uploads/2021/05/dragon-ball.jpg",
          "https://i.pinimg.com/originals/4e/aa/b6/4eaab69fcf8d928738072cd355a980db.jpg",
          "https://static.zerochan.net/Uzumaki.Naruto.full.3291213.jpg",
          "https://bbts1.azureedge.net/images/p/full/2021/02/cc97eea8-abbf-47f5-a66a-652c6feacdc7.jpg"
          ]

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():

  #url = request.form.name['image_upload']
  url = request.form['image_upload']
  #pred_image = urllib.request.urlretrieve(url,"local-img.jpg")
  val = predict_image(url,VGG_model,pca_reload)
  im_show = images[val]
  output = lable_encode[val]
  
  return render_template('result.html',im_show=im_show, prediction_text='Predicted hero is  {}'.format(output))

@app.errorhandler(500)
def wrong_url(e):
  return render_template('error.html')



if __name__ == "__main__":
  app.run(host='0.0.0.0')