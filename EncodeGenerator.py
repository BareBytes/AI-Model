import cv2
import face_recognition
import pickle
import os
import tkinter as tk

# Importing the Guard images to a list
# Murtaza's youtube
folderPath = 'Images'
PathList = os.listdir(folderPath)
print(PathList)
imageList = []
GuardId = []
for path in PathList:
    imageList.append(cv2.imread(os.path.join(folderPath,path)))
    GuardId.append(os.path.splitext(path)[0])
    #print(os.path.splitext(path)[0])
print(GuardId)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList



print("Encoding Started...")
encodeListKnown = findEncodings(imageList)
encodeListKnownWithIds = [encodeListKnown, GuardId]
print("Encoding completed")

file = open("EncodeFile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")
