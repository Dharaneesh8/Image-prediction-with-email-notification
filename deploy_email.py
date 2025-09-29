import cv2
import numpy as np
import smtplib
import time
from email.message import EmailMessage
from keras.models import load_model

# Load model and labels
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Set up camera
camera = cv2.VideoCapture(0)
np.set_printoptions(suppress=True)

# Email config
EMAIL_ADDRESS = "dharaneeshj8@gmail.com"       # Your email
EMAIL_PASSWORD = "ybvxfiyltlwlqozy"           # Use app password if 2FA is enabled
TO_EMAIL = "asgaurdsaviour@gmail.com"             # Recipient email

# Helper function to send email
def send_email(predicted_class):
    msg = EmailMessage()
    msg.set_content(f"The model has detected: {predicted_class}")
    msg['Subject'] = f"Object Detected: {predicted_class}"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"Email sent: {predicted_class}")
    except Exception as e:
        print("Failed to send email:", e)

# Track time of last notification
last_sent_time = 0
notify_interval = 300  # 5 minutes in seconds

while True:
    ret, image = camera.read()
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow("Webcam Image", image)

    # Prepare image
    image_np = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1

    # Prediction
    prediction = model.predict(image_np)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Output prediction
    print("Class:", class_name, "Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Email every 5 minutes
    current_time = time.time()
    if current_time - last_sent_time > notify_interval:
        send_email(class_name)
        last_sent_time = current_time

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
