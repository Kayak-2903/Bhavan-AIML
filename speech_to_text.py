# Python program to translate
# speech to text and text to speech
# import pyAudio
import speech_recognition as sr
import pyttsx3 
import text2emotion as te
# Initialize the recognizer import numpy as np
import wavio

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
		with sr.AudioFile(r'D:\KKFiles\Final year project\output.wav') as source2:
			print("in with")
			# wait for a second to let the recognizer
			# adjust the energy threshold based on
			# the surrounding noise level 
			# r.adjust_for_ambient_noise(source2, duration=0.2)
			print("after removing ambient noise")
			#listens for the user's input 
			audio2 = r.record(source2)
			# audio2 = 
			print(audio2)
			print("listening")
			# Using google to recognize audio
			MyText = r.recognize_google(audio2)
			# print("Did you say "+MyText)
			# SpeakText(MyText)
			emotions = te.get_emotion(MyText)
			# print(emotions)
	except sr.RequestError as e:
		print("Could not request results; {0}".format(e))
		
	except Exception as e:
		print("unknown error occured; {0}".format(e))
