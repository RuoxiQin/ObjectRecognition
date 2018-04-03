# imports

import numpy as np
import cv2
import timeit
import os.path
import random
# from feed_forward import blackbox

# cropping script 
def img_crop(img, dim = (227, 227), loc = None):
    if loc == None:
        a = img.shape
        loc = [(0,0), (a[0] - 1, a[1] - 1)]
    crop_img = img[loc[0][0]:loc[1][0], loc[0][1]:loc[1][1]]
    resized_image = cv2.resize(crop_img, dim)
    return resized_image 
    
    
def blackbox(x,y):
    return None

def locate_object_contours(img_obj, img_scene):
    divide = 3
    contours = []
    
    img_scene_dim = img_scene.shape
    seg_increment = int((min(img_scene_dim[0], img_scene_dim[1]) - 1) / divide)
    
    for k in range(divide):
        contour = []
        current_seg = (k+1) * seg_increment
        shift_increment = int(current_seg / 3)
        scan_x = int((img_scene_dim[0] - seg_increment - 1) / shift_increment)
        scan_y = int((img_scene_dim[1] - seg_increment - 1) / shift_increment)

        for i in range(scan_x):
            sub_contour = []
            for j in range(scan_y):
                tl = (i * shift_increment, j * shift_increment)
                test_scene = \
                    img_crop(img_scene, \
                            loc = \
                            [(tl[0], tl[1]), 
                             (tl[0]+current_seg-1,tl[1]+current_seg-1 )])
                sub_contour.append(blackbox(img_obj, test_scene))
            contour.append(subcontour)
        contours.append(np.array(contour))
    return contours
