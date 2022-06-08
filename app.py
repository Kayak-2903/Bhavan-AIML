from flask import Flask, request
import traceback
import firebase

app = Flask(__name__)

@app.route('/emotion-detection', methods=['POST'])
def emotion_detect():
    from emotion_analysis import emotion_detection
    audioFile = request.files['audioFile']
    # audioFile = body['audioFile']
    audioFile.save('output.wav')
    return emotion_detection(open('output.wav', 'rb'))

@app.route('/sign-up', methods=['POST'])
def sign_up():
    user = request.get_json()
    try:
        return firebase.addUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/login', methods=['POST'])
def login():
    login_cred = request.get_json()
    try:
        return firebase.login(login_cred)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/update-user', methods=['PUT'])
def updateUser():
    user = request.get_json()
    try:
        return firebase.updateUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/delete-user', methods=['DELETE'])
def deleteUser():
    user = request.get_json()
    try:
        return firebase.deleteUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

@app.route('/get-user', methods=['POST'])
def getUser():
    user = request.get_json()
    try:
        return firebase.getUser(user)
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        print(e)
        return "Failed"

if __name__=='__main__':
    app.run(debug=True)