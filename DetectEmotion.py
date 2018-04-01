#!/usr/bin/env python3

import numpy as np
from collections import Counter
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.ndimage import zoom
import cv2

# define constants
UPDATE_THRESHOLD = 3    # adjust this to define how fast the prediction label changes

model = model_from_json(open('./models/Face_model_architecture.json').read())
model.load_weights('./models/Face_model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# openCV setup
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

def extract_face_features(gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face

def detect_face(frame):
        faceCascade = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(48, 48)
            )
        return gray, detected_faces

prediction_list = []
last_prediction = 8

while True:

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # detect faces
    gray, detected_faces = detect_face(frame)
    face_index = 0

    # predict output
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # draw rectangle around face 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (66, 244, 240), 4)
            
            # extract features
            extracted_face = extract_face_features(gray, face, (0.075, 0.05))

            # predict smile
            prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))

            # draw extracted face in the top right corner
            frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # buffer for detected emotions
            if len(prediction_list) < UPDATE_THRESHOLD:
                prediction_list.append(int(prediction_result))
                prediction_result = last_prediction
            else:
                # use the most common predicted emotion from the last UPDATE_THRESHOLD predictions
                prediction_result = int(str(Counter(prediction_list).most_common(1))[2])
                prediction_list = []

            # annotate main image with a label
            if prediction_result == 0:
                cv2.putText(frame, "Angry",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 10)
            elif prediction_result == 1:
                cv2.putText(frame, "Disgust",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (37, 237, 180), 10)
            elif prediction_result == 2:
                cv2.putText(frame, "Fear",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)
            elif prediction_result == 3:
                cv2.putText(frame, "Happy!!",(x,y), cv2.FONT_ITALIC, 2, (0, 255, 0), 10)
            elif prediction_result == 4:
                cv2.putText(frame, "Sad",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (191, 0, 95), 10)
            elif prediction_result == 5:
                cv2.putText(frame, "Surprise",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 229, 0), 10)
            else:
                cv2.putText(frame, "Neutral",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 10)

            # increment counter
            face_index += 1

            last_prediction = prediction_result

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
