from xml.dom.minidom import Attr
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('serviceAccountKey.json')

firebase_admin.initialize_app(cred)

db = firestore.client()

def addUser(user):
    present = db.collection('users').where('email', '==', user['email']).stream()
    if len(list(present)) > 0:
        return 'Email Already Exists'
    db.collection('users').add({
        'firstName': user['firstName'],
        'lastName': user['lastName'],
        'email': user['email'],
        'password': user['password']
    })
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