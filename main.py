import cv2
import easyocr
import numpy as np

img = cv2.imread('car_img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

plates = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

results = plates.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
print(results)

for (x, y, w, h) in results:
    text = easyocr.Reader(['ru'])
    text = text.readtext(gray[y:y+h, x:x+w])
    res = text[0][-2]
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, res , (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)


cv2.imshow('Result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()