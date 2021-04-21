# Import all the necessary files!
import os
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
import base64
import cv2
import json
import ast
import numpy as np
from PIL import Image

graph = tf.get_default_graph()

app = Flask(__name__)
CORS(app)
config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.InteractiveSession(config=config)
set_session(sess)

model = load_model('facenet_keras.h5',compile=False)

def img_to_encoding(path, model, flag=True):
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

#database = {}
#database["Adil"] = img_to_encoding("images/1618963301.jpg", model, True)
#database["laughingAdil"] = img_to_encoding("images/1618963301.jpg", model)
#print(database)

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

@app.route('/register', methods=['POST'])
def register():
    print("Performing Registration")
    img_name = str(int(datetime.timestamp(datetime.now())))
    print("Register request 1 ", request.form)
    print("Register request 2 ", request.values)
    print("Register request 3 ", request.files['picture'])
    
    #img_data = request.get_json()['fd']
    img_data = request.files['picture']
    img = Image.open(img_data)
    print(img.size) 
    img.save('images/'+img_name+'.jpg')
    #with open('images/'+img_name+'.jpg', "wb") as fh:
    #    fh.write(base64.b64decode(img_data[22:]))
    path = 'images/'+img_name+'.jpg'
    
    encodings = img_to_encoding(path, model)
    return json.dumps({"status":200, "encodings": str(encodings)})

def who_is_it(image_path, database, model):
    print("Enetered in who is it method")
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
    print(type(arr), len(arr))
    arr = ast.literal_eval(arr[0])
    
    database = {}
    
    for i in range(len(arr)):
        uid = arr[i]["userId"]
        if(len(arr[i]) >= 2):
            encodings = ast.literal_eval(arr[i]["encodings"])
            database[uid] = encodings
    
    return database

@app.route('/verify', methods=['POST'])
def change():
    
    req_dict = request.form
    req_dict = req_dict.to_dict(flat=False)
    req_dict = req_dict["userInfo"]
    
    database = create_db(req_dict)
    #print(database)
    #print("Register request 2 ", request.files['picture'])

    
    img_data = request.files['picture']
    print(img_data)
    img_name = str(int(datetime.timestamp(datetime.now())))
    img = Image.open(img_data)
    print(img.size) 
    img.save('images/'+img_name+'.jpg')
    
    path = 'images/' + img_name +'.jpg'
    print(path)
    min_dist, identity = who_is_it(path, database, model)
    print(min_dist, identity)
    os.remove(path) 
    if min_dist > 5:
        return json.dumps({"status": 404})
    
    return json.dumps({"status": 200, "id": identity})
    


if __name__ == "__main__":
    app.run()