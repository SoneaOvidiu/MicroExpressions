import cv2
import numpy as np
import dlib
import LengthCalculator as lc
from Emotions import Emotions as em
#import UnsupervisedLearning as ul

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)


def draw_landmarks(image,landmarks):
    coords=[]
    for index in range(68):
        coords.append((landmarks.part(index).x,landmarks.part(index).y))
    coords=np.array(coords,np.int32)
    for elems in coords:
        cv2.circle(image,(elems[0],elems[1]), 1, (0,255,0),-1)
    return frame,coords


font = cv2.FONT_HERSHEY_PLAIN
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emot=em()
#model,dicti=ul.create_model()
while True:
    _, frame = cap.read()
    frame=adjust_gamma(frame,1.7)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        if len(faces)==1:
            landmarks = predictor(gray, face)
            frame,coords=draw_landmarks(frame,landmarks)
            #print(lc.printing(frame,coords))
            '''file=open('ratio_values1.csv','a')
            for elem in lc.printing(frame,coords):
                file.write(str(elem)+', ')
            file.write("happy\n")
            file.close()'''
            value=lc.printing(frame,coords)
            value.pop(0)
            value.pop(0)
            #ul.get_prediction(model,dicti,value)
            emot.process_data(lc.printing(frame,coords))
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 113:
        break

cap.release()
cv2.destroyAllWindows()