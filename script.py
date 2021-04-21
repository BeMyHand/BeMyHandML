# Import all the necessary files!
import os
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
from PIL import Image
import cv2
import json
import ast
import numpy as np


graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.InteractiveSession(config=config)
set_session(sess)

model = load_model('facenet_keras.h5',compile=False)

def img_to_encoding(path, model, flag=True):
    '''
    params: path to an image
            model used to calculate encodings
            flag true in case of registration
            flag false in case of verification
    
    returns: image encodings
    '''
    img1 = cv2.imread(path, 1)
    img = img1[...,::-1]
    dim = (160, 160)
      # resize image
    if(img.shape != (160, 160, 3)):
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    x_train = np.array([img])
    embeddings = model.predict_on_batch(x_train)
    
    if(flag):
        print("for registration")
        arr = []
        for enc in embeddings[0]:
            arr.append(enc)
        return arr
    return embeddings

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

@app.route('/register', methods=['POST'])
def register():
    '''
    recieves --> a user image
    returns a json response with status code
    and user image encodings
    '''
    print("Performing Registration")
    img_name = str(int(datetime.timestamp(datetime.now())))
    
    img_data = request.files['picture']
    img = Image.open(img_data)
    print(img.size) 
    img.save('images/'+img_name+'.jpg')
    
    path = 'images/'+img_name+'.jpg'
    
    encodings = img_to_encoding(path, model)
    return json.dumps({"status":200, "encodings": str(encodings)})

def who_is_it(image_path, database, model):
    '''
    params: path to the image
            database, a dictionary which contains user id
            along with it's encodings
    returns: minimum distance and user Id found from database
    '''
    encoding = img_to_encoding(image_path, model, False)
    min_dist = 100
    
    # Loop over the database dictionary's ids and encodings.
    for (uid, db_enc) in database.items():   
        dist = np.linalg.norm(encoding-db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = uid

    if min_dist > 5:
        print("Not in the database.")
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        
    return min_dist, identity

def create_db(arr):
    '''
    params: recieves and array with encodings and user ids with noise data
    returns: a database dictionary of encodings against each user id
    '''
    arr = ast.literal_eval(arr[0])
    
    database = {}
    
    for i in range(len(arr)):
        uid = arr[i]["userId"]
        if(len(arr[i]) >= 2):
            encodings = ast.literal_eval(arr[i]["encodings"])
            database[uid] = encodings
    
    return database

@app.route('/verify', methods=['POST'])
def verify():
    '''
    request --> recieves a json response which contains encodoings stored in
                database and user image used for verification
    response --> returns json response with status code and 
                user id in case user is found   
    '''
    req_dict = request.form
    req_dict = req_dict.to_dict(flat=False)
    req_dict = req_dict["userInfo"]
    
    database = create_db(req_dict)
    
    img_data = request.files['picture']
    img_name = str(int(datetime.timestamp(datetime.now())))
    img = Image.open(img_data)
    img.save('images/'+img_name+'.jpg')
    
    path = 'images/' + img_name +'.jpg'
    min_dist, identity = who_is_it(path, database, model)
    print(min_dist, identity)
    os.remove(path) 
    if min_dist > 5:
        return json.dumps({"status": 404})
    
    return json.dumps({"status": 200, "id": identity})
    


if __name__ == "__main__":
    app.run()