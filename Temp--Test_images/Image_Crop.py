import cv2
import numpy as np
import timeit

######### 
def crop(img, dim = (227, 227), loc = None):
    
    if loc == None:
        a = img.shape
        loc = [(0,0), (a[0] - 1, a[1] - 1)]
    
    crop_img = img[loc[0][0]:loc[1][0], loc[0][1]:loc[1][1]]
    resized_image = cv2.resize(crop_img, dim)
    
    # Display New Image
    #~ cv2.imshow("New Image", resized_image)
    #~ cv2.waitKey(0)
    
    return resized_image 



########

if __name__ == '__main__':
    
    ### path
    
    path = 'Original/homeoffice/scene_8/'
    #~ path = 'Original/bedroom/scene_31/'
    #~ path = 'Original/kitchen/scene_23/'
    
    output_path = 'Generated/set-1/'
    
    ### file
    filename = '2.JPG'
    #~ filename = 'scene.jpg'
    
    ### counting segmentations
    count = 0
    
    ### Begin timer
    start = timeit.default_timer()
    
    img = cv2.imread(path + filename)
    
    ### cropping 
    #~ to_crop = None
    to_crop = [(370,550), (630,730)]
    new_image = crop(img, loc = to_crop)
    
    if count > 0:
        filename = filename[:-4] + '-' + str(count) + '.jpg'
    
    cv2.imwrite(output_path + filename, new_image)
    
    ### End timer
    
    stop = timeit.default_timer()

    print('Execution time: ' + str(stop - start))
