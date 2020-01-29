import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImamgemComId():
    # path of the pictures taken
    ways = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    face = []
    ids = []

    for caminhoImg in ways:
        faceImg = cv2.cvtColor(cv2.imread(caminhoImg), cv2.COLOR_BGR2GRAY)
        # get id of each photo
        id = int(os.path.split(caminhoImg)[-1].split('.')[1])
        ids.append(id)
        face.append(faceImg)
    return np.array(ids), face


ids, faces = getImamgemComId()
# all recognizers
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print('Training completed')