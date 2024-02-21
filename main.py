import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
font=cv2.FONT_HERSHEY_COMPLEX
# Load the model
model = load_model("waste_segregation_model.h5")


def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break


    cv2.imshow('Video Feed', frame)


    img_array = preprocess_image(frame)

    
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    classes = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper",
               "plastic", "shoes", "trash", "white-glass"]
    predicted_class = classes[index]
    print(f"The predicted image is: {predicted_class}")
    img=cv2.putText(frame,predicted_class,(30,70),cv2.FONT_HERSHEY_COMPLEX,1.3,(25,50,40),3)
    cv2.imshow('Video Feed', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
