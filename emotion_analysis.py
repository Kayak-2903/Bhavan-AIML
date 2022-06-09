# import warnings
# warnings.filterwarnings('ignore')
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
# import matplotlib.pyplot as plt
# import librosa.display
# import logging
# logging.getLogger('tensorflow').disabled = True


# def waveplot(data, sr, emotion):
#     plt.figure(figsize=(10,4))
#     plt.title(emotion, size=20)
#     librosa.display.waveshow(data, sr=sr)
#     plt.show()
    
# def spectogram(data, sr, emotion):
#     x = librosa.stft(data)
#     xdb = librosa.amplitude_to_db(abs(x))
#     plt.figure(figsize=(10,4))
#     plt.title(emotion, size=20)
#     librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
#     plt.colorbar()
    

def extract_mfcc(filename):
    print(filename)
    import librosa
    y, sr = librosa.load(filename, duration = 3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 40).T, axis = 0)
    return mfcc

def load_dataset():
    paths = []
    labels = []
    import os
    for dirname, _, filenames in os.walk('archive'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            labels.append(filename.split('_')[-1].split('.')[0].lower())
    print('Dataset is loaded')
    
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, Dropout
    model = Sequential([
        SimpleRNN(123, return_sequences=False, input_shape=(40,1)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax'),
    ])
    import pandas as pd
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels
    df.head()
    df['label'].value_counts()

    return df

def retrain(emotion):
    import pandas as pd
    df = pd.DataFrame()
    paths = []
    paths.append('retrain.wav')
    labels = []
    labels.append(emotion.lower())
    df['speech'] = paths
    df['label'] = labels
    df.head()
    df['label'].value_counts()
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = []
    X = [X_mfcc]
    X = np.array(X)
    X = np.expand_dims(X, -1)
    
    
    from keras.models import load_model
    new_model = load_model('new_model.h5')
    # new_model = model_from_json(open("new_model.json", 'r').read())
    # new_model.load_weights("new_model.h5")    

    # new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    new_model.summary()
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']])

    y = y.toarray()
    new_model.fit(X, y, epochs=100, batch_size=512, shuffle=True)
    new_model.save('new_model.h5')

def createInputExpectedOutput(df):
    print('in create expectedoutput')
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    print('done in create expectedoutput')
    X = []
    X = [x for x in X_mfcc]
    X = np.array(X)
    X = np.expand_dims(X, -1)
    
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']])

    y = y.toarray()
    print('done y encoding')

    return X, y

def train_model(X, y):
    print('in train model')
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, Dropout
    model = Sequential([
        SimpleRNN(123, return_sequences=False, input_shape=(40,1)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax'),
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print('in model fit')
    model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512, shuffle=True)
    # json_file = model.to_json()
    # with open('model.json', 'w') as file:
    #     file.write(json_file)
    # model.save_weights('model.h5')
    model.save('model.h5')
    return model


def predict_voice_emotion(model):
    path = np.array(['output.wav'])[0]
    X_mfcc = extract_mfcc(path)

    X = [X_mfcc]
    X = np.array(X)
    X = np.expand_dims(X, -1)
    predicted_emotion = model.predict(X)
    total_sum = np.sum(predicted_emotion, dtype=float)
    # print(total_sum)
    emotions=['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Sad']
    dict = {}
    for i in range(7):
        dict[emotions[i]] = predicted_emotion.item(i) / total_sum
    return dict

def predict_text_emotion():
    # Python program to translate
    # speech to text and text to speech
    # import pyAudio
    import speech_recognition as sr
    import text2emotion as te
    # Initialize the recognizer import numpy as np

    # wa = wavio.read("output.wav")  #Read a .wav file
    # print("x= "+str(wa.data))   #Data
    # print("rate= "+str(wa.rate))    #Rate
    # print("sampwidth= "+str(wa.sampwidth))  #sampwidth
    # wavio.write("output.wav", wa.data, wa.rate,sampwidth = wa.sampwidth)   #Error is here
    # Function to convert text to
    # # speech
    # def SpeakText(command):
        
    # 	# Initialize the engine
    # 	engine = pyttsx3.init()
    # 	engine.say(command) 
    # 	engine.runAndWait()
        
    r = sr.Recognizer()
    # Loop infinitely for user to
    # speak
    flag = True
    while(flag): 
        flag = False	
        # Exception handling to handle
        # exceptions at the runtime
        try:
            
            # use the microphone as source for input.
            with sr.AudioFile(open('output.wav', 'rb')) as source2:
                # print("in with")
                # wait for a second to let the recognizer
                # adjust the energy threshold based on
                # the surrounding noise level 
                # r.adjust_for_ambient_noise(source2, duration=0.2)
                # print("after removing ambient noise")
                #listens for the user's input 
                audio2 = r.record(source2)
                # audio2 = 
                # print(audio2)
                # print("listening")
                # Using google to recognize audio
                MyText = r.recognize_google(audio2)
                print("Did you say "+MyText)
                # SpeakText(MyText)
                # emotions = te.get_emotion(MyText)
                import joblib
                model = joblib.load(open("notebooks/emotion_classifier_pipe_lr_03_june_2021.pkl","rb")) 
                # print(emotions)
                new_text = remove_not(MyText, model)
                print(new_text)
                # print(MyText)
                predicted = model.predict_proba([new_text])
                emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Shame', 'Surprise']
                dict_1 = {}

                for i in range(8):
                    dict_1[emotions[i]] = predicted[0][i]

                dict_2 = te.get_emotion(new_text)

                ratio_1 = 1
                ratio_2 = 1

                dict = {}

                for emotion in dict_2:
                    dict[emotion] = (ratio_1 * dict_1[emotion] + ratio_2 * dict_2[emotion])/2
                for emotion in dict_1:
                    if emotion not in dict_2:
                        dict[emotion] = ratio_1 * dict_1[emotion]

                
                return dict_2, MyText
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except Exception as e:
            print("unknown error occured; {0}".format(e))

def remove_not(text, model):
    import re
    dict = {
        'not [a-z ]* angry': 'ok',
        'not [a-z ]* sad': 'ok',
        'not [a-z ]* happy': 'sad',
        'not [a-z ]* ashamed': 'ok',
        'not [a-z ]* surprised': 'ok',
        'not [a-z ]* scared': 'brave',
        'not [a-z ]* disgusted': 'ok',
        'stupid[a-z ]*': 'angry',
        'joy[a-z ]*': 'happy',
        'annoy[a-z ]*': 'angry',
        'how dare you': 'angry',
        'no way': 'angry',
        'mad': 'angry',
        'not [a-z ]* accept[a-z ]*':'angry',
        'oh my god': 'surprise',
        'not [a-z ]* joy[a-z ]*': 'sad',
        'not [a-z ]* annoy[a-z ]*': 'ok',
    }

    for key in dict:
        text = re.sub(key, dict[key], text)
    return text

def predict_combined_emotion(predicted_voice, predicted_text):
    ratio_text = 1
    ratio_voice = 1
    dict = {}
    for emotion in predicted_text:
        dict[emotion] = ratio_text * predicted_text[emotion] + ratio_voice * predicted_voice[emotion]
    for emotion in predicted_voice:
        if emotion not in predicted_text:
            dict[emotion] = ratio_voice * predicted_voice[emotion]
    return dict

def emotion_detection(file):
    try:
        # from keras.models import model_from_json
        # model = model_from_json(open("model.json", 'r').read())
        # model.load_weights("model.h5")
        from keras.models import load_model
        model = load_model('model.h5')
    except:
        
        df = load_dataset()

        X, y = createInputExpectedOutput(df)

        model = train_model(X, y)

        

    predicted_voice = predict_voice_emotion(model)
    predicted_text, text = predict_text_emotion()
    predict_combined = predict_combined_emotion(predicted_voice, predicted_text)
    print('voice:',predicted_voice)
    print('text:',predicted_text)
    print('combined:',predict_combined)
    max_value = 0
    predicted_emotion = ''
    for emotion in predict_combined:
        if(max_value < predict_combined[emotion]):
            predicted_emotion = emotion
        max_value = max(max_value, predict_combined[emotion])
    print(predicted_emotion)
    from flask import jsonify
    return jsonify(
        predicted_voice_map = predicted_voice,
        predicted_text_map = predicted_text,
        predicted_combined_map = predict_combined,
        predicted_emotion_value = predicted_emotion,
        predicted_text_value = text
    )

# print(emotion_detection(open('output.wav', 'rb')))