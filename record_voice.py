# import sounddevice as sd
# from scipy.io.wavfile import write

# fs = 44100  # Sample rate
# seconds = 5  # Duration of recording
# startspeaking = input("Start Speaking? Click any key")
# myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
# sd.wait()  # Wait until recording is finished
# write('output.wav', fs, myrecording)  # Save as WAV file 

import pyaudio
import wave

audio = pyaudio.PyAudio()

stream = audio.open(format = pyaudio.paInt16, channels = 1, rate = 44100, input = True, frames_per_buffer=1024)
frames = []
try:
    while(True):
        print("started reading")
        data= stream.read(1024)
        frames.append(data)
except KeyboardInterrupt:
    pass
stream.stop_stream()
stream.close()
audio.terminate()

sound_file = wave.open("output.wav", "wb")
sound_file.setnchannels(1)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(44100)
sound_file.writeframes(b''.join(frames))
sound_file.close()