from PIL import Image, ImageOps
import glob
import os


os.chdir(r"face_recognaization\Take-Picture\\")
all_images = glob.glob("*.png")
counter = 0
for image in all_images:
    size = (48, 48)
    img = Image.open(image)

    fit_and_resized_image = ImageOps.fit(img, size, Image.ANTIALIAS)
    
    fit_and_resized_image.save("image-3-"+str(counter)+'.jpg')
    counter +=1

all_images = glob.glob("*.jpg")
#all_images.append(glob.glob("*.jpg"))
for image in all_images:
    size = (48, 48)
    img = Image.open(image)

    fit_and_resized_image = ImageOps.fit(img, size, Image.ANTIALIAS)
    
    fit_and_resized_image.save("image-3-"+str(counter)+'.jpg')
    counter +=1