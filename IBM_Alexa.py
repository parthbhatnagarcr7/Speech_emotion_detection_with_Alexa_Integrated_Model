import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes

import torch
import librosa
import numpy as np

# ------------------------------------------------
# 1. Load your trained PyTorch model
# ------------------------------------------------
MODEL_PATH = "model_weight.pth"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()  # set to evaluation mode


# ------------------------------------------------
# 2. Preprocessing function (same as before)
# ------------------------------------------------
def preprocess_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    return torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0)


# ------------------------------------------------
# 3. Define a function to record + detect emotion
# ------------------------------------------------
def detect_emotion():
    """
    1) Record audio from the microphone using speech_recognition.
    2) Save it as a temporary WAV file.
    3) Use the PyTorch model to predict the emotion.
    4) Return the emotion label (e.g., "happy", "sad", etc.).
    """
    listener = sr.Recognizer()
    temp_filename = "temp.wav"

    # Record audio
    with sr.Microphone() as source:
        print("Listening for emotion detection...")
        audio_data = listener.listen(source)

    # Save the audio to a WAV file
    with open(temp_filename, "wb") as f:
        f.write(audio_data.get_wav_data())

    # Preprocess and predict
    features = preprocess_audio(temp_filename)
    with torch.no_grad():
        prediction = model(features)
        emotion_idx = torch.argmax(prediction, dim=1).item()

    # Map index to label
    emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']
    predicted_emotion = emotion_labels[emotion_idx]
    return predicted_emotion


# ------------------------------------------------
# 4. Alexa-like Voice Assistant Code
# ------------------------------------------------
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


def talk(text):
    engine.say(text)
    engine.runAndWait()


def take_command():
    listener = sr.Recognizer()
    command = ""
    try:
        with sr.Microphone() as source:
            print('listening...')
            voice = listener.listen(source)
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'alexa' in command:
                command = command.replace('alexa', '')
                print("User command:", command)
    except:
        pass
    return command


def run_alexa():
    command = take_command()
    print("Received command:", command)

    if 'play' in command:
        song = command.replace('play', '')
        talk('playing ' + song)
        pywhatkit.playonyt(song)

    elif 'time' in command:
        time = datetime.datetime.now().strftime('%I:%M %p')
        talk('Current time is ' + time)

    elif 'who the heck is' in command:
        person = command.replace('who the heck is', '')
        info = wikipedia.summary(person, 1)
        print(info)
        talk(info)

    elif 'date' in command:
        talk('sorry, I have a headache')

    elif 'are you single' in command:
        talk('I am in a relationship with wifi')

    elif 'joke' in command:
        talk(pyjokes.get_joke())

    # -----------------------------------------------
    #  **Detect Emotion** if user asks
    # -----------------------------------------------
    elif 'emotion' in command or 'mood' in command:
        talk("Let me check how you're sounding. Please speak after the beep.")
        emotion = detect_emotion()
        response = f"I think you're sounding {emotion}."
        print(response)
        talk(response)

    else:
        talk('Please say the command again.')


# ------------------------------------------------
# 5. Main Loop
# ------------------------------------------------
if __name__ == '__main__':
    while True:
        run_alexa()
