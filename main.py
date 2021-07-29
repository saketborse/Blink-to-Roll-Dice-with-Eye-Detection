from time import sleep

from cv2 import FONT_HERSHEY_SIMPLEX
import numpy as np
import cv2
import diceroll as dr

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye = cv2.CascadeClassifier('eye.xml')
first_read = True
cap = cv2.VideoCapture(0)
ret, img = cap.read()
d = dr.dice()
while (ret):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 1, 1)
    faces = face.detectMultiScale(gray, 1.3, 5, minSize=(200, 200))

    if(len(faces)>0):
        for(x, y, w, h) in faces:
            img = cv2.rectangle(
                img, (x, y), (x+w, y+h), (0, 255, 0), 2
            )
            roi_fa = gray[y:y+h, x:x+w]
            roi_fa_clr = img[y:y + h, x:x + w]
            eyes = eye.detectMultiScale(roi_fa, 1.3, 5, minSize=(50, 50))
            if(len(eyes)>=2):

                if(first_read):
                    cv2.putText(img,'Blink', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
                else:
                    cv2.putText(img, 'EYE not Found', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
            else:
                d = dr.dice()

                if (first_read ):

                    cv2.putText(img, d, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

                    cv2.waitKey(1000)
                else:
                    cv2.waitKey(3000)
                    first_read = True
    else:
        cv2.putText(img, 'nf', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)


    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord("s") and first_read):
        first_read = False
cap.release()
cv2.destroyAllWindows()