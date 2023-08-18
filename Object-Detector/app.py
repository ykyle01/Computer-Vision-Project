# OpenCV program to detect cat face in real time 
# import libraries of python OpenCV 
# where its functionality resides 
import cv2
from PIL import Image
import requests
from io import BytesIO
import numpy
  
imageUrl = "https://firebasestorage.googleapis.com/v0/b/visionproject-4299b.appspot.com/o/images%2F1d242e3f-2668-4d88-922c-0a8bcb9fd74c?alt=media&token=2746aa0a-588d-4999-888c-2d0c771b9c99"

# load the required trained XML classifiers 
# https://github.com/Itseez/opencv/blob/master/ 
# data/haarcascades/haarcascade_frontalcatface.xml 
# Trained XML classifiers describes some features of some 
# object we want to detect a cascade function is trained 
# from a lot of positive(faces) and negative(non-faces) 
# images. 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml') 

# Get iamge from imageUrl
response = requests.get(imageUrl)
pil_image = Image.open(BytesIO(response.content)).convert('RGB') 
pil_image = pil_image.resize((500,500))
open_cv_image = numpy.array(pil_image) 
# Convert RGB to BGR 
img = open_cv_image[:, :, ::-1].copy() 

while 1:
  # convert to gray scale of each frames 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

  # Detects faces of different sizes in the input image 
  faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

  for (x,y,w,h) in faces: 
      # To draw a rectangle in a face 
      cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) 
      roi_gray = gray[y:y+h, x:x+w] 
      roi_color = img[y:y+h, x:x+w] 


  # Display an image in a window 
  cv2.imshow('img', img)

  # Wait for Esc key to stop 
  k = cv2.waitKey(30) & 0xff
  if k == 27: 
      break

# De-allocate any associated memory usage 
cv2.destroyAllWindows()