from flask import Flask, request
import traceback
from emotion_analysis import emotion_detection
import firebase_connection
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/emotion-detection', methods=['POST'])
def emotion_detect():
    from emotion_analysis import emotion_detection
    audioFile = request.files['audioFile']
    email = request.form['email']
    # audioFile = body['audioFile']
    audioFile.save(email+'_detect.wav')
    firebase_connection.getModel(email)
    output = emotion_detection(email+'_detect.wav', email+'.h5')
    import os
    os.remove(email+'.h5')
    os.remove(email+'_detect.wav')
    return output

@app.route('/sign-up', methods=['POST'])
def sign_up():
    user = request.get_json()
    try:
        return firebase_connection.addUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/reset-dataset', methods=['PUT'])
def reset():
    email = request.form['email']
    try:
        return firebase_connection.resetDataset(email)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/retrain', methods=['POST'])
def retrain():
    audioFile = request.files['audioFile']
    email = request.form['email']
    emotion = request.form['emotion']
    audioFile.save(email+'_retrain.wav')
    try:
        return firebase_connection.retrain(email, emotion.lower())
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/add-dataset', methods=['POST'])
def addDataset():
    audioFile = request.files['audioFile']
    name = request.form['name'].lower()
    path = "archive/set/TESS Toronto emotional speech set data/"+name+"/"+audioFile.filename
    audioFile.save(path)
    return emotion_detection(path)

@app.route('/login', methods=['POST'])
def login():
    login_cred = request.get_json()
    try:
        return firebase_connection.login(login_cred)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/update-user', methods=['PUT'])
def updateUser():
    user = request.get_json()
    try:
        return firebase_connection.updateUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/delete-user', methods=['DELETE'])
def deleteUser():
    user = request.get_json()
    try:
        return firebase_connection.deleteUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/get-user', methods=['POST'])
def getUser():
    user = request.get_json()
    try:
        return firebase_connection.getUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

if __name__=='__main__':
    app.run(debug=True)