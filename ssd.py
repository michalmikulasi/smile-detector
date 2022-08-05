
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import cv2 as cv


# create datafreame
df = pd.read_xml('C:/Users/micha/OneDrive/Počítač/Homework/haarcascade_eye.xml')
df.head()
#another dataframe
df2 = pd.read_xml('C:/Users/micha/OneDrive/Počítač/Homework/haarcascade_smile.xml')
df2.head()

#another df
df3 = pd.read_xml('C:/Users/micha/OneDrive/Počítač/Homework/haarcascade_frontalface_default.xml')
df3.head()

#detecting faces
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_smile.xml')


cap = cv.VideoCapture(0)
#capturing video from webcam
while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_grey = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        smile = smile_cascade.detectMultiScale(roi_grey, 1.7, 22)
        for (sx,sy,sw,sh) in smile:
            cv.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh), (0,0,255), 2)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



#release the camera
cap.release()
cv.destroyAllWindows()

#generating documentation
def get_pydoc_text(module):
    "Returns pydoc generated output as text"
    doc = pydoc.TextDoc()
    loc = doc.getdocloc(pydoc_mod) or ""
    if loc:
        loc = "\nMODULE DOCS\n    " + loc + "\n"

    output = doc.docmodule(module)

    # cleanup the extra text formatting that pydoc preforms
    patt = re.compile('\b.')
    output = patt.sub('', output)
    return output.strip(), loc

