# imports

import os.path
import numpy as np
import cv2
import random
import xml.etree.ElementTree as ET


# Init parameters

path_prefix = '../CS640_Project_dataset/'
dir_prefix = 'scene_'
obj_suffix = '.JPG'
file_scene = 'scene.jpg'
file_xml = 'scene.xml'
obj_dim = 500

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
    
    
    
    
    
# get obj from xml 
def bound_xml(tup, path):
    path_to_xml = path + ikea_scenes[tup[0]] + '/' \
            + dir_prefix + str(tup[1]) + '/' \
            + file_xml
    target_obj = tup[2] 
    
    # XML parsing
    found = False 
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
            break
        else: 
            continue 
    if Found:
        return min(x_list), max(x_list), min(y_list), max(y_list)
    else: 
        return 0,0,0,0

# backtrack path 
def backtrack_tuple(tup, path):
    return (path + ikea_scenes[tup[0]] + '/' \
            + dir_prefix + str(tup[1]) + '/' \
            + str(tup[2]) + obj_suffix)

# cropping script 
def img_crop(img, dim = (227, 227), loc = None):
    if loc == None:
        a = img.shape
        loc = [(0,0), (a[0] - 1, a[1] - 1)]
    crop_img = img[loc[0][0]:loc[1][0], loc[0][1]:loc[1][1]]
    resized_image = cv2.resize(crop_img, dim)
    return resized_image 


##### training data generation ######

def gen_object(target_object, tup):
    return_imgs = []
    
    # backtracking path 
    t_p = backtrack_tuple(t_p_tuple, path_prefix) 
    
    o_img = cv2.imread(t_p) 
    # gen cropped obj's
    for i in range(2):
        r_4 = [random.getrandbits(1) for j in range(4)]
        return_imgs.append( \
            img_crop(o_img, loc = \
            [(r_4[0], r_4[1]), \
            (obj_dim - 1 - r_4[2], obj_dim - 1 - r_4[3])] \
            ))
        
        #### Testing codes
        # cv2.imwrite(str(i)+'_test.jpg', return_imgs[i])
    return return_imgs

def gen_scene_1(target_object, bounds):
    
    return [np.array([])]

##### INPUT FUNC #####

def input_func():
    # choose target objects randomly
    n = 3574
    target_obj = random.sample(range(1,n+1),20)
    
    # init
    features = {'object': [], 'scene': []}
    labels = []
    
    for x in target_obj:
        #### get xml, otherwise reroll until found 
        get_xml = bound_xml(x, path_prefix)
        while get_xml == (0,0,0,0):
            x = random.randint(1, n)
            get_xml = bound_xml(x, path_prefix)
        
        obj_tuple = backtrack_path(x, path_prefix) 
        #### objects 
        obj_list = gen_object(x, obj_tuple)
        features['objects'].extend(obj_list)
        
        #### 
        fts, ls = gen_object(x, obj_tuple) # 50 dummy 1's
        features['scenes'].extend(fts) 
        for i in range(25):
            labels.append(1)
    
    return features, labels



    
