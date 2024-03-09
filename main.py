import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import serial

ser=serial.Serial('COM3',9600)
font=cv2.FONT_HERSHEY_COMPLEX
model = load_model("model\waste_segregation_model_v2(100% accuracy).h5")


def preprocess_image(img):
    img = cv2.resize(img, (150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    reduced_image=cv2.resize(frame,(150,150))
    if not ret:
        print("Failed to capture image")
        break


    cv2.imshow('Video Feed', frame)


    img_array = preprocess_image(reduced_image)

    
    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    classes = ["leather","metal pecies","paper","plastic"]
    predicted_class = classes[index]
    print(f"The predicted image is: {predicted_class}")
    img=cv2.putText(frame,predicted_class,(30,70),cv2.FONT_HERSHEY_COMPLEX,1.3,(25,50,40),3)
    cv2.imshow('Video Feed', img)
    
    if predicted_class=="leather":
        ser.write(b'0')
    if predicted_class=="metal pecies": 
        ser.write(b'1')
    if predicted_class=="paper": 
        ser.write(b'2')        
    if predicted_class=="plastic": 
        ser.write(b'3')
    else:
        ser.write(b'4')
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
