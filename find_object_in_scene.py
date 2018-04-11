# imports

import numpy as np
import cv2
import timeit
import os.path
import random
# importing our model 
from neuralNet import Detector


# parameters 
path_to_model = "./tmp/original"
path_scan_prefix = "./scan_images/set-1/"



# cropping script 
def img_crop(img, dim = (227, 227), loc = None):
    if loc == None:
        a = img.shape
        loc = [(0,0), (a[0] - 1, a[1] - 1)]
    crop_img = img[loc[0][0]:loc[1][0], loc[0][1]:loc[1][1]]
    resized_image = cv2.resize(crop_img, dim)
    return resized_image 
    
def scene_dissect(img_scene, divide = 3):
    segments = []
    img_segments = []
    num_segments = 0
    
    img_scene_dim = img_scene.shape
    seg_increment = int((min(img_scene_dim[0], img_scene_dim[1]) - 1) / divide)
    
    for k in range(divide):
        current_seg = (k+1) * seg_increment
        shift_increment = int(current_seg / 3)
        scan_x = int((img_scene_dim[0] - seg_increment - 1) / shift_increment)
        scan_y = int((img_scene_dim[1] - seg_increment - 1) / shift_increment)
        
        # append to segments
        segments.append([scan_x, scan_y])
        
        for i in range(scan_x):
            for j in range(scan_y):
                for r in range(5):
                    # random shift
                    r_4 = [random.getrandbits(1) for j in range(4)]
                    
                    tl = (i * shift_increment, j * shift_increment)
                    test_scene = \
                        img_crop(img_scene, \
                                loc = \
                                [(tl[0] + r_4[0], tl[1] + r_4[1]), 
                                 (tl[0] + current_seg - 1 - r_4[2],
                                 tl[1] + current_seg - 1 - r_4[3])])
                    img_segments.append(test_scene)
                    num_segments += 1
                
    return img_segments, segments, num_segments
    

def object_dissect(img_object):
    obj_dim = 500
    r_4 = [random.getrandbits(1) for j in range(4)]
    return img_crop(img_object, loc = [(r_4[0], r_4[1]),             
            (obj_dim - 1 - r_4[2], obj_dim - 1 - r_4[3])])
        

def segment_images(img_object, img_scene, divide = 3):
    
    features = {'objects': [], 'scenes': []}
    
    # scene
    scene_seg, segments, num_segments = scene_dissect(img_scene)
    
    # object
    for i in range(num_segments):
        features['objects'].append(object_dissect(img_object))
    
    return features, segments

if __name__ == "__main__":
    
    # the model
    detector =  Detector(path_to_model)
    
    # load scene/object
    path_to_object = path_scan_prefix + "object.jpg"
    path_to_scene = path_scan_prefix + "scene.jpg"
    img_object = cv2.imread(path_to_object)
    img_scene = cv2.imread(path_to_scene)
    
    # image dissection
    feed_features, segments = segment_images(img_object, img_scene)
    
    
    # predicts 
    feed_features = \
        [scene_dissect(), object_dissect()]
    feed_features["objects"] = \
        np.array(feed_features["objects"]).astype(np.float32)
    feed_features["scenes"] = \
        np.array(feed_features["scenes"]).astype(np.float32)
    feed_data = {"features": feed_features}
    
    predictions = detector.predict(features, labels)
    
    contours = []
    for seg in segments:
        contour = np.zeros((seg[0],seg[1]))
        for i in range(seg[0]):
            for j in range(seg[1]):
                avg = 0
                for r in range(5):
                    avg += predictions.pop(0)
                avg = avg / 5
                contour[i][j] = avg
        print(contour)
        contours.append(contour)

    
    
# EOF
