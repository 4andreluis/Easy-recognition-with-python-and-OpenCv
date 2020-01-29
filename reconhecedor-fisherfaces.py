import cv2

detectorFace = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read('classificadorFisher.yml')
width, height = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while True:
    connected, imagem = camera.read()
    grayImg = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    detectedFaces = detectorFace.detectMultiScale(grayImg,
                                                  scaleFactor=1.5,
                                                  minSize=(30, 30))
    for (x, y, l, a) in detectedFaces:
        faceImg = cv2.resize(grayImg[y:y + a, x:x + l], (width, height))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confidence = recognizer.predict(faceImg)
        name = ''
        if id == 1:
            name = 'Andre'
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (255, 0, 0), 2)
            cv2.putText(imagem, name, (x, y + (a + 30)), font, 2, (255, 0, 0))
        else:
            name = 'Unknown'
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv2.putText(imagem, name, (x, y + (a + 30)), font, 2, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()