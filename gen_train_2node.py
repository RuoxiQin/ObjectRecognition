#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The script to generate training data.
The neuralNet.py needs the input_func here as input_func in TensorFlow API
"""

# imports
import os.path
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET


# Init parameters

# The directory where the training data is stored
path_prefix = './dataset/'
dir_prefix = 'scene_'
obj_suffix = '.JPG'
file_scene = 'scene.jpg'
file_xml = 'scene.xml'
obj_dim = 500
# Some objects are occluded in the scene. The label of those objects can be set 
# as different number other than 0 or 1. Here we still set it as 1
occ_para = 1.0

# The batch size 
epoch_size = 100 # must be a multiple of 50

# The list of folders contained by the training directory
ikea_scenes = [
    'bathroom', 'bedroom', 'childrenroom', 
    'hallway', 'homeoffice', 'kitchen', 
    'laundry', 'livingroom', 'outdoor']


# compute the dataset_structure

def scan_num_obj(path):
    i = 0
    if not os.path.exists(path + '/' + file_xml):
        return i
    while True:
        fname = path + '/' + str(i+1) + obj_suffix
        if os.path.exists(fname):
            i += 1
        else:
            break
    return i

def scan_scene(path_to_scene):
    count = []
    i = 1
    while True:
        path_name = path_to_scene + '/' + dir_prefix + str(i)
        i += 1 
        if os.path.exists(path_name):
            count.append(scan_num_obj(path_name))
        else:
            break
    return count

def scan_all(path_to_data):
    output = []
    for x in ikea_scenes:
        output.append(scan_scene(path_to_data + '/' + x))
    return output

dataset_structure = scan_all(path_prefix)

# backtrack the path to the objects

def backtrack_path(target_num, path_to_data):
    scene_loc = 0
    scene_num = 0
    found = False
    for x in dataset_structure:
        for y in x:
            scene_num = 0
            if (target_num <= y):
                found = True
                break
            else: 
                target_num -= y
                scene_num += 1
        if found:
            break
        scene_loc += 1
    return (scene_loc, scene_num + 1, target_num)


## GENERATE TRAINING PICS

# get obj from xml 
def bound_xml(tup, path):
    path_to_xml = path + ikea_scenes[tup[0]] + '/'             + dir_prefix + str(tup[1]) + '/'             + file_xml
    target_obj = tup[2] 
    # XML parsing
    Found = False 
    occluded = False
    tree = ET.parse(path_to_xml)
    root = tree.getroot()
    for obj in root.iter('object'):
        if(obj.find('deleted').text == "0") and obj.find('name').text == str(target_obj):
            x_list = []
            y_list = []
            for pt in obj.iter('pt'):
                x = int(pt.find('x').text)
                y = int(pt.find('y').text)
                x_list.append(x)
                y_list.append(y)
            
            Found = True # Found!
            if obj.find('occluded').text == 'yes':
                occluded = True
            break
        else: 
            continue 
    if Found:
        return (min(x_list), max(x_list), min(y_list), max(y_list)), occluded
    else: 
        return (0,0,0,0), occluded

# backtrack path 
def backtrack_tuple_obj(tup, path):
    return (path + ikea_scenes[tup[0]] + '/'             + dir_prefix + str(tup[1]) + '/'             + str(tup[2]) + obj_suffix)

def backtrack_tuple_scene(tup, path):
    return (path + ikea_scenes[tup[0]] + '/'             + dir_prefix + str(tup[1]) + '/'             + file_scene)

# cropping script 
def img_crop(img, dim = (227, 227), loc = None):
    if loc == None:
        a = img.shape
        loc = [(0,0), (a[0] - 1, a[1] - 1)]
    crop_img = img[loc[0][0]:loc[1][0], loc[0][1]:loc[1][1]]
    resized_image = cv2.resize(crop_img, dim)
    return resized_image 


##### training data generation ######

def gen_object(tup):
    return_imgs = []
    
    # backtracking path 
    t_p = backtrack_tuple_obj(tup, path_prefix) 
    
    o_img = cv2.imread(t_p) 
    # gen cropped obj's
    for i in range(50):
        r_4 = [random.getrandbits(1) for j in range(4)]
        return_imgs.append(             img_crop(o_img, loc =             [(r_4[0], r_4[1]),             (obj_dim - 1 - r_4[2], obj_dim - 1 - r_4[3])]             ))
        
        #### Testing codes
        # cv2.imwrite(str(i)+'_test.jpg', return_imgs[i])
    return return_imgs

def gen_scene_1(tup, bounds):
    return_imgs = []
    
    # parse image 
    t_p = backtrack_tuple_scene(tup, path_prefix)
    o_scene = cv2.imread(t_p)
    o_scene_dim = o_scene.shape
    sq_dim = min(
            int(max(bounds[1]-bounds[0], bounds[3]-bounds[2]) * 1.2), 
            min(o_scene_dim[0], o_scene_dim[1]))
    _tl = [int((bounds[2]+bounds[3] - sq_dim)/ 2), 
               int((bounds[0]+bounds[1] - sq_dim)/ 2)]
    tl = [max(_tl[0] - max(0, _tl[0] + sq_dim - o_scene_dim[0]), 0),
            max(_tl[1] - max(0, _tl[1] + sq_dim - o_scene_dim[1]), 0)]
    n_scene = img_crop(
                o_scene, dim = (sq_dim, sq_dim), loc = \
                [(tl[0],tl[1]),(tl[0] + sq_dim - 1,tl[1] + sq_dim - 1)])
    for i in range(25):
        r_4 = [random.randint(0,5) for j in range(4)]
        return_imgs.append(             img_crop(n_scene, loc =             [(r_4[0], r_4[1]),             (sq_dim - 1 - r_4[2], sq_dim - 1 - r_4[3])]             ))
        # cv2.imwrite('Temp/' + str(i)+'_test.jpg', return_imgs[i])
    return return_imgs

def gen_scene_0(tup, bounds):
    return_imgs = []
    
    # parse next scene
    if tup[1] < len(dataset_structure[tup[0]]):
        new_tup = (tup[0], tup[1] + 1, tup[2])
    else:
        new_tup = (tup[0], 1, tup[2])
    
    # parse image 
    t_p = backtrack_tuple_scene(new_tup, path_prefix)
    o_scene = cv2.imread(t_p)
    o_scene_dim = o_scene.shape
    increment = int((min([o_scene_dim[0], o_scene_dim[1]])-1) / 5)
    for i in range(1,6):
        for j in range(1,6):
            tl = (random.randint(0, o_scene_dim[0] - i * increment - 1), 
                  random.randint(0, o_scene_dim[1] - i * increment - 1))
            br = (tl[0] + i * increment - 1, tl[1] + i * increment - 1)
            return_imgs.append(img_crop(o_scene, loc = [tl, br]))
    return return_imgs

##### INPUT FUNC #####

def input_func():
    # choose target objects randomly
    n = 3574
    target_obj = random.sample(range(1,n+1), int(epoch_size / 50))
    
    # init
    features = {'objects': [], 'scenes': []}
    labels = []
    
    for x in target_obj:
        #### get xml, otherwise reroll until found 
        obj_tuple = backtrack_path(x, path_prefix)
        get_xml, occluded = bound_xml(obj_tuple, path_prefix)
        while get_xml == (0,0,0,0):
            x = random.randint(1, n)
            obj_tuple = backtrack_path(x, path_prefix) 
            get_xml, occluded = bound_xml(obj_tuple, path_prefix)
        
        #### objects 
        obj_list = gen_object(obj_tuple)
        features['objects'].extend(obj_list)
        
        #### scenes
        ## 1
        fts = gen_scene_1(obj_tuple, get_xml) # 25
        features['scenes'].extend(fts) 
        for i in range(25):
            if occluded:
                labels.append(np.array([[occ_para],[1-occ_para]]))
            else:
                labels.append(np.array([[1],[0]]))
        ## 0
        fts = gen_scene_0(obj_tuple, get_xml) # 25
        features['scenes'].extend(fts) 
        for i in range(25):
            labels.append(np.array([[0],[1]]))
    
    return features, labels

"""
features, labels = input_func()
print(len(features['objects']))
print(len(features['scenes']))
print(len(labels))
"""

