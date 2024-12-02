import os,cv2
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import glob

'''
import sys
sys.path.append('/binhe/ml-ex')

from importlib import reload
from imp import reload
import mylibs 
try:
    reload(mylibs.face_utils)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
from mylibs.face_utils import imShow,imShowPlt,show_video
'''

# Source code for autocrop: https://github.com/leblancfg/autocrop

# import cv2
# import numpy as np

#--------------------------------Head detection-------------------------------------
def mtcnn_face_detect(img_file, pads = (0,0,0,0), one_face_only = True):
    # assert sum(pads)>0 , "there should be pad exsit"

    if type(img_file) == str:
        img = cv2.imread(img_file)
    else:
        img = img_file
                
    # print("mtcnn_face_detect: original shape", img.shape)
    
     # Convert to grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
     # Detect the faces
    faces, _ = mtcnn.detect(img_rgb, landmarks=False)
    
    results = []
    
    # Draw the rectangle around each face
#     if ( faces and faces!= []):

    try:
        if faces.any():
            for face in faces:
                try:
        #             x, y, w, h = int(faces[0][0]), int(faces[0][1]), int(faces[0][2])-int(faces[0][0]), int(faces[0][3])-int(faces[0][1])
        #             face_region = img_rgb[y-delta:y+h+delta, x-delta:x+w+delta, :]
        #             return face_region, [x, y, w, h]

                    x1, y1, x2, y2 = int(face[0]), int(face[1]), int(face[2]), int(face[3])

                    y1 = max(0, y1 - pads[0])
                    y2 = min(img_rgb.shape[0], y2 + pads[1])
                    x1 = max(0, x1 - pads[2])
                    x2 = min(img_rgb.shape[1], x2 + pads[3])
                    face_region = img_rgb[y1:y2, x1:x2, :]

                    results.append([face_region, (y1, y2, x1, x2), (int(face[1]), int(face[3]), int(face[0]), int(face[2]))])
                except:
                    print("face incorrect")
        else:
            print('can not detect face, use original image')
            return [img_rgb, (0,img_rgb.shape[0],0,img_rgb.shape[1]), (0,img_rgb.shape[0],0,img_rgb.shape[1])]

        if one_face_only:
            return results[0]
        else:
            return results
    except:
        print('0 can not detect face, use original image')
        return [img_rgb, (0,img_rgb.shape[0],0,img_rgb.shape[1]), (0,img_rgb.shape[0],0,img_rgb.shape[1])]