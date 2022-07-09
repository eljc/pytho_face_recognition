import numpy as np
import cv2
import json
import face_recognition
import os

image = face_recognition.load_image_file("static/images/*.jpg")
face_encoding = face_recognition.face_encodings(image)[0]

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye_tree_eyeglasses.xml')

listaEncodeConhecido = [
    face_encoding
]

print('Encoding Complete')

def predict(test_image, threshold, uploadWidth, uploadHeight):
    texto_imagem = ''
    nomeImagens = ['Elder', 'Aline']

    imgS = cv2.resize(test_image, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    output = []

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    item = Object()
    item.version = "0.0.1"
    item.numObjects = len(facesCurFrame)
    item.threshold = threshold
    output.append(item)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(listaEncodeConhecido, encodeFace)
        faceDis = face_recognition.face_distance(listaEncodeConhecido, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = nomeImagens[matchIndex].upper()
            # print(name)
            top, right, bottom, left = faceLoc
            #cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            texto_imagem = name
            print('Reconhecido: ', name)

            eyes = eye_cascade.detectMultiScale(test_image, scaleFactor=1.2, minNeighbors=4)


        # Add some metadata to the output
        item = Object()
        item.class_name = "{}".format(texto_imagem)
        item.name = texto_imagem
        item.score = 1
        # item.x = float(1 - left/uploadWidth)
        # item.y = float(1 - top/uploadHeight)
        # item.height = float((bottom-top)/uploadHeight)
        # item.width = float((right-left)/uploadWidth)
        item.x = left
        item.y = top
        item.height = bottom - top
        item.width = right - left

        output.append(item)

    outputJson = json.dumps([ob.__dict__ for ob in output])
    return outputJson


# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name = "Face recognition demo using TensorFlow REST API"

    def toJSON(self):
        return json.dumps(self.__dict__)