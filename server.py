from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
#system level operations (like loading files)
import sys
#for reading operating system data
import os
app = Flask(__name__)
model = load_model('saved_model.h5')
@app.route("/")
def hello_world():
    return "Ini API Model Deploy Tourismo"

@app.route("/predictimage", methods=['POST'])
def image_predict():
    # Print the request data
    print('Request Method:', request.method)
    print('Request Headers:', request.headers)

    # # Check if image file is present in the request
    # if 'image' not in request.files:
    #     return 'No image file found in the request.'

    # Load and process the image file
    image_file = request.files['image']
    print('Image File Name:', image_file.filename)
    print('Image Content Type:', image_file.content_type)
    image = Image.open(image_file)
    image = image.resize((150, 150))
    imagearr = img_to_array(image)
    imagearr = np.expand_dims(imagearr, axis=0)
    images = np.vstack([imagearr])
    

    # Perform prediction using the loaded model
    prediction = model.predict(images, batch_size=10)
    output = np.argmax(prediction)
    lokasi  = ""
    if output == 0:
        lokasi = 'Candi Borobudur'
    elif output == 1:
        lokasi = "Garuda Wisnu Kencana"
    elif output == 2:
        lokasi = "Monas"
    elif output == 3:
        lokasi = "Candi Prambanan"
    elif output == 4:
        lokasi = "Danau Toba"
    else:
        lokasi = "lokasi tidak ada di database"

    # Return the predicted class label as a response
    return f'Lokasi: {lokasi}'
