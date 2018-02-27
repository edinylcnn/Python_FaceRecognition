# -*- coding: cp1254 -*-
import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer\\trainingData.yml")
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 0, 255)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(150,255,255),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf <= 50):
            if(Id==1):
                Id="Edin (ADMIN)"
            elif(Id==2):
                Id="Edin (ADMIN)"
            elif(Id==3):
                Id="Edin (ADMIN)"
            elif(Id==4):
                Id="Edin (ADMIN)"
            elif(Id==5):
                Id="Nadir"
            elif(Id==6):
                Id="Nadir"
            elif(Id==7):
                Id="Edin"
            elif(Id==8):
                Id="Ermin"
            elif(Id==9):
                Id="MuratVuranok"

            
        else:
            Id="Unknown"
        #cv2.cv.PutText(cv2.fromarray(im),str(Id), (x,y+h), font, 255)
        cv2.putText(im, str(Id), (x,y+h), fontface, fontscale, fontcolor)
    cv2.imshow("im", im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
