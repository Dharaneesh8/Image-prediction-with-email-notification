# Image-prediction-with-email-notification
This project demonstrates how to use Google Teachable Machine with Python to build an image recognition system. After prediction, the system can notify the user in two ways:
Email Notification – sends an email with the prediction result every 5 seconds.
Voice Notification – uses text-to-speech to announce the predicted class aloud.

STEP 1 – Train Model in Teachable Machine
Open Teachable Machine.
Click Get Started.
Select Image Project or Audio Project.
Choose Standard Image Model.

STEP 2 – Add Classes & Train
Add your classes (e.g., Cat, Dog, Car).
Upload images or collect samples with webcam.
Click Train Model.

STEP 3 – Export Model
Click Export Model.
Choose Keras (.h5) format.
Download the .zip file.

STEP 4 – Extract Files
After extraction, you’ll get:
keras_model.h5 – trained model file
labels.txt – class labels

STEP 5 – Setup Project
Place your Python script in the same folder as:
keras_model.h5
labels.txt
