import cv2
import numpy as np
cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera")

img_counter = 133
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = cam.read()
    frame2=frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)    
    cv2.imshow("Camera", frame2)
    if not ret:
        break
    k = cv2.waitKey(1)
    
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "Take-Picture\\name_1_{}.png".format(img_counter)
        for i, face in enumerate(detector.detectMultiScale(frame)):
            x, y, w, h = face
            sub_face = frame[y:y + h, x:x + w]
            cv2.imwrite(img_name, sub_face)
        print("{} written!".format(img_name))
        img_counter += 1
cam.release()
cv2.destroyAllWindows()