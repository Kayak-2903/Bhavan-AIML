import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import emotion_analysis
config = {
  'apiKey': "AIzaSyCEV_k7uy2aMIuOeMcPP4QbB6lmFjpXs-w",
  'authDomain': "bhavana-e7d97.firebaseapp.com",
  'databaseURL': "https://bhavana-e7d97-default-rtdb.firebaseio.com",
  'projectId': "bhavana-e7d97",
  'storageBucket': "bhavana-e7d97.appspot.com",
  'messagingSenderId': "1093189440195",
  'appId': "1:1093189440195:web:fdd31500cfce57a57f8876",
  'measurementId': "G-T6CWEBZSXJ"
}

import pyrebase
pyrebaseInitialize = pyrebase.initialize_app(config)
storage = pyrebaseInitialize.storage()



cred = credentials.Certificate('serviceAccountKey.json')

firebase_admin.initialize_app(cred, {
    'storageBucket': 'gs://bhavana-e7d97.appspot.com'
})

db = firestore.client()

def getModel(email):
    storage.child('userFiles/' + email + '/model/model.h5').download(email+'.h5')
    # storage.child('userFiles/' + email + '/model/X.csv').download(email+'.csv')

def retrain(email, emotion):
    count = 0
    print(emotion, email)
    storage.child('userFiles/' + email + '/model/X.csv').download(email+'_X.csv')
    storage.child('userFiles/' + email + '/model/y.csv').download(email+'_y.csv')
    emotion_analysis.retrain(emotion.lower(), email)
    from datetime import datetime
    path_on_cloud = 'userFiles/' + email + '/' + emotion + '/' + datetime.now().strftime("%d-%m-%Y-T%H:%M:%S") + '.wav'
    path_local = email+'_retrain.wav'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + email + '/model/model.h5'
    path_local = email+'_retrain.h5'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + email + '/model/X.csv'
    path_local = email+'_X.csv'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + email + '/model/y.csv'
    path_local = email+'_y.csv'
    storage.child(path_on_cloud).put(path_local)
    import os
    os.remove(email+'_retrain.h5')
    os.remove(email+'_retrain.wav')
    os.remove(email+'_X.csv')
    os.remove(email+'_y.csv')
    return "success"

def resetDataset(email):
    path_on_cloud = 'userFiles/' + email + '/model/model.h5'
    path_local = 'model.h5'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + email + '/model/X.csv'
    path_local = 'X.csv'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + email + '/model/y.csv'
    path_local = 'y.csv'
    storage.child(path_on_cloud).put(path_local)
    return "Success"

def addUser(user):
    present = db.collection('users').where('email', '==', user['email']).stream()
    if len(list(present)) > 0:
        return 'Email Already Exists'
    db.collection('users').add(user)
    path_on_cloud = 'userFiles/' + user['email'] + '/model/model.h5'
    path_local = 'model.h5'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + user['email'] + '/model/X.csv'
    path_local = 'X.csv'
    storage.child(path_on_cloud).put(path_local)
    path_on_cloud = 'userFiles/' + user['email'] + '/model/y.csv'
    path_local = 'y.csv'
    storage.child(path_on_cloud).put(path_local)
    return "Success"

def login(login_cred):
    present = db.collection('users').where('email', '==', login_cred['email']).where('password', '==', login_cred['password']).stream()
    if len(list(present)) > 0:
        return 'Success'
    return "Invalid Credentials"    

def updateUser(user):
    result = db.collection('users').where('email', '==', user['email']).get()
    if len(result) == 0:
        return "Email doesn't exist"
    doc = db.collection('users').document(result[0].id)
    doc.update(user)
    return "Success"

def deleteUser(user):
    result = db.collection('users').where('email', '==', user['email']).get()
    if len(result) == 0:
        return "Email doesn't exist"
    doc = db.collection('users').document(result[0].id)
    doc.delete()
    return "Success"

def getUser(user):
    result = db.collection('users').where('email', '==', user['email']).get()
    if len(result) == 0:
        return "Email doesn't exist"
    from flask import jsonify
    return jsonify(result[0].to_dict())