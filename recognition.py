import face_recognition
import os
import csv
import cv2
import numpy as np
import math

from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    face_ages = []
    face_genders = []
    known_face_encodings = []
    known_face_names = []
    known_face_ages = []
    known_face_genders = []
    process_current_frame = True
    face_classifier = cv2.CascadeClassifier(r'/Users/anggaarindra/Downloads/MachineLearning/EmotionDetection/haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    classifier = load_model(r'/Users/anggaarindra/Downloads/MachineLearning/EmotionDetection/model.h5')

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        with open('/Users/anggaarindra/Downloads/MachineLearning/FaceRecognition/dataset.csv', 'r') as file:
            csv_reader = csv.DictReader(file, delimiter=';')
            for row in csv_reader:
                name = row['nama']
                image_path = row['path']
                age = row['umur']
                gender = row['jenis_kelamin']

                face_image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ages.append(age)
                self.known_face_genders.append(gender)
                print(self.known_face_names)

    def detect_emotion(self, frame):
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = self.classifier.predict(roi)[0]
                label = self.emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                labels.append(label)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return labels

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame)

                self.face_names = []
                self.face_ages = []
                self.face_genders = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    age = "Unknown"
                    gender = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        age = self.known_face_ages[best_match_index]
                        gender = self.known_face_genders[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')
                    self.face_ages.append(age)
                    self.face_genders.append(gender)

            self.process_current_frame = not self.process_current_frame

            # Detect emotions
            emotion_labels = self.detect_emotion(frame)

            # Display the results
            for (top, right, bottom, left), name, age, gender, emotion_label in zip(self.face_locations, self.face_names,
                                                                                    self.face_ages, self.face_genders,
                                                                                    emotion_labels):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name, age, gender, and emotion
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
                cv2.putText(frame, f'Age: {age}', (left + 6, bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)
                cv2.putText(frame, f'Gender: {gender}', (left + 6, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)
                cv2.putText(frame, f'Emotion: {emotion_label}', (left + 6, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognition()
    fr.run_recognition()

