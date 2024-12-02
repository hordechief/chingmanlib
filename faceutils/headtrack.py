# Source code for autocrop: https://github.com/leblancfg/autocrop

import cv2
import numpy as np

#--------------------------------Head detection-------------------------------------
def img_crop(img_file, mtcnn, delta=0):
    if type(img_file) == str:
        img = cv2.imread(img_file)
    else:
        img = img_file
     # Convert to grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect the faces
    faces, _ = mtcnn.detect(img_rgb, landmarks=False)
    # Draw the rectangle around each face
    if ( faces!= []):
        # for (x, y, w, h) in faces:
        try:
            x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2]), int(faces[0][3])
            # cv2.rectangle(img, 
            #             (x, y), 
            #             (w, h), 
            #             (255, 0, 0), 
            #             2)
            # face crop and do your thing
            # delta = 100
            face_region = img_rgb[y-delta:h+delta, x-delta:w+delta, :]

            return face_region, [x, y, w, h]
        except:
            print('can not detect face, use original image')
            return img_rgb, []

