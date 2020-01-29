import cv2
import numpy as np

classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeSorter = cv2.CascadeClassifier('haarcascade-eye.xml')
camera = cv2.VideoCapture(0)
amostra = 1
id = str(input('Enter your ID: '))
width, height = 220, 220
print('You need 25 photos')
print('Press "q" to capture the image')
print('Capturing face...')
while True:
    connected, imagem = camera.read()
    grayImg = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # colocar imagem em cinza

    # Detected faces
    detectedFaces = classifier.detectMultiScale(grayImg,
                                                scaleFactor=1.5,
                                                minSize=(150, 150))
    # Put the rectangle on the face
    for x, y, l, a in detectedFaces:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # Detected eyes
        region = imagem[y:y + a, x:x + l]
        grayEyeRegion = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        detectedEyes = eyeSorter.detectMultiScale(grayEyeRegion)
        for ox, oy, ol, oa in detectedEyes:
            cv2.rectangle(region, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2)
            # Light restriction and key definition for photo capture
            if cv2.waitKey(1) & 0xFF == ord('q') and np.average(grayImg) > 110:
                imagemFace = cv2.resize(grayImg[y:y + a, x:x + l], (width, height))
                # save photo in the "fotos" folder with the name "pessoa.{id}.amostra.jpg"
                cv2.imwrite(f'fotos/pessoa.{id}.{str(amostra)}.jpg', imagemFace)
                print(f'Photo {str(amostra)} successfully captured')
                amostra += 1
    cv2.imshow('Face', imagem)
    cv2.waitKey(1)
    if amostra > 25:
        break
print('Photo captured successfully')
camera.release()
cv2.destroyAllWindows()
