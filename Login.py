import os
import pickle
import numpy as np
from ultralytics import YOLO
import cv2
import face_recognition
import time
import datetime
import tkinter as tk
from tkinter import messagebox

# Load the YOLO model
model = YOLO('yolov10n.pt')

# Load a sample picture and learn how to recognize it.
srishti_image = face_recognition.load_image_file("Images/17111998.jpg")
srishti_face_encoding = face_recognition.face_encodings(srishti_image)[0]
# Load a second sample picture and learn how to recognize it.
alia_image = face_recognition.load_image_file("Images/665588.jpg")
alia_face_encoding = face_recognition.face_encodings(alia_image)[0]
# Known face encodings and names (for demonstration purposes)
known_face_encodings = [
    srishti_face_encoding,
    alia_face_encoding
]
known_face_names = [
    "Srishti",
    "Alia"
]

login_name = None
login_face_encoding = None


def face_login():
    global login_name, login_face_encoding
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    authenticated = False
    attempt = 0

    while not authenticated and attempt < 3:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all face locations and face encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            # See if the face is a match for any known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                login_name = known_face_names[best_match_index]
                login_face_encoding = known_face_encodings[best_match_index]
                # login_name.append(name)
                authenticated = True
                break

        if authenticated:
            tk.messagebox.showinfo("Login", f"Welcome, {login_name}!")

        else:
            attempt += 1
            tk.messagebox.showerror("Login Failed", "Face not recognized. Please try again.")
            if attempt < 3:
                tk.messagebox.showinfo("Retry", "Attempting again...")

            else:
                tk.messagebox.showerror("Login Failed", "Maximum attempts reached. Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                return False

    # cap.release()
    # cv2.destroyAllWindows()
    return authenticated


# Main application flow
if face_login():
    if login_name:

        # Continue with the rest of the application
        # Webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 520)
        cap.set(4, 620)

        # Importing the mode images to a list
        folderModePath = 'Images'
        modePathList = os.listdir(folderModePath)
        imageModeList = []
        GuardId = []
        for path in modePathList:
            imageModeList.append(cv2.imread(os.path.join(folderModePath, path)))

        person_class_id = next(key for key, value in model.names.items() if value == 'person')
        ret = True

        # Dictionary to store the timing
        person_timing = {}
        person_name = {}

        # initialize the video capture
        start_time = time.time()
        Login_time = datetime.datetime.today()
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Load the encoding File
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        encodeListKnown, GuardId = encodeListKnownWithIds

        # read frames
        while ret:
            ret, frame = cap.read()

            # resizing the image
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            faceCuFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCuFrame)

            current_time = time.time() - start_time

            # detect the object
            results = model.track(frame, persist=True)

            # only include person
            person_results = [res for res in results[0].boxes if int(res.cls) == person_class_id and res.conf > 0.5]

            # update timing for each person
            for person in person_results:
                if person.id is None:
                    continue

                is_logged_in_person = False
                for encodeFace in encodeCurFrame:
                    matches = face_recognition.compare_faces([login_face_encoding], encodeFace)
                    if matches[0]:
                        is_logged_in_person = True

                if is_logged_in_person:
                    person_id = int(person.id)
                    if person_id in person_timing:
                        person_timing[person_id]['last_seen'] = current_time
                    else:
                        person_timing[person_id] = {'first_seen': current_time, 'last_seen': current_time}

                    # Draw person ID and duration on the frame
                    first_seen = person_timing[person_id]['first_seen']
                    last_seen = person_timing[person_id]['last_seen']
                    duration = last_seen - first_seen
                    x1, y1, x2, y2 = map(int, person.xyxy[0].tolist())
                    text = f'ID: {person_id}, Name: {login_name}, Time: {duration:.2f}s'
                    cv2.putText(frame, text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 25), 3)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logff = datetime.datetime.today()
                break

        cap.release()
        cv2.destroyAllWindows()

        # Calculate the time for each person in the frame
        # last_seen = {}
        total_time = 0
        for person_id, times in person_timing.items():
            # if person_name == login_name:
                # name = str(person_name[person_id])
            total_time += times['last_seen'] - times['first_seen']
            # name = str(person_name[person_id])
            # print(f'Person {person_id}: {login_name} was in the frame for {total_time:.2f} seconds')
                # name = str(person_name[person_id])
                # if person_name[person_id] not in last_seen:
                    # last_seen[person_id] = total_time
                    # print(f'Person {person_id}: {login_name} was in the frame for {total_time:.2f} seconds')
                # else:
                    # last_seen[person_id] += total_time
                    # print(f'Person {person_id}: {login_name} was in the frame for {total_time:.2f} seconds')

            # totalTime = sum(list(last_seen.values()))

        print(f'{login_name} was in the frame for {total_time:.2f} seconds')
        print(f'Login Date and Time: {Login_time}')
        print(f'Logged off Date and Time: {logff}')


    else:
        print('Logged user is not the same')
else:
    print("Exiting application due to failed authentication.")