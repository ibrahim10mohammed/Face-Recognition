import os
import cv2

image_path = "ibrahim"


def save_faces(cascade, imgname):
    img = cv2.imread(os.path.join(image_path, imgname))
    for i, face in enumerate(cascade.detectMultiScale(img)):
        x, y, w, h = face
        sub_face = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join("crop-ibrahim", "{}_{}.jpg".format(imgname, i)), sub_face)

if __name__ == '__main__':
    face_cascade = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(face_cascade)
    # Iterate through files
    for f in [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]:
        save_faces(cascade, f)

counter = 0
path = os.chdir("crop-picture name")
filenames = os.listdir(path)
for filename in filenames:
    counter+=1
    os.rename(filename, filename.replace(filename,"picture name"+str(counter))+'.jpg')