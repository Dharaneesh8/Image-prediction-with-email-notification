import cv2
import numpy as np
import threading
from keras.models import load_model
from gtts import gTTS
from playsound import playsound
import os
import time

# Load model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Initialize camera
camera = cv2.VideoCapture(0)

# Track prediction and timing
last_prediction = ""
last_speech_time = 0

def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "temp_audio.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)
    except Exception as e:
        print("Audio error:", e)

print("🔊 Using Google Text-to-Speech (gTTS)")


while True:
    # Capture frame
    ret, image = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize and preprocess
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_normalized = (image_resized.astype(np.float32) / 127.5) - 1
    image_input = np.expand_dims(image_normalized, axis=0)

    # Prediction
    prediction = model.predict(image_input, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    predicted_label = class_name[2:].strip()

    # Display label on screen
    display_text = f"{predicted_label} ({confidence_score*100:.1f}%)"
    cv2.putText(image, display_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Waste Classifier", image)

    # Speak only if prediction changed and confidence is high
    if predicted_label != last_prediction and confidence_score > 0.85:
        last_prediction = predicted_label
        last_speech_time = time.time()
        threading.Thread(target=speak_text, args=(f"This is {predicted_label}",)).start()

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
