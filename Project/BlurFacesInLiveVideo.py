import cv2
import numpy as np


#Blur function
def blurFaces(image):
    imig = image
    faces = detectFaces(image)
    for face in faces:
        x, y, w, h = face
        imig[y:y + h, x:x + w] = cv2.GaussianBlur(image[y:y + h, x:x + w], (25, 25), 7)

    return img


#Detect face function
def detectFaces(image):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray_img)
    return faces


capture = cv2.VideoCapture(0)
while True:
    _, img = capture.read()

    blurred = blurFaces(img)

    cv2.imshow("LiveBlur", blurred)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
