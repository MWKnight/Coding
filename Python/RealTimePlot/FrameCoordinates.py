import numpy as np


def GetFrameCoordinates(img):

    height, width = np.shape(img[:,:,0])

    middle_height = round(height/2.0)
    middle_width = round(width/2.0)
    
    ep = 0.05

    ################# Find top of frame ################

    found_top = False
    
    for k in range(height):
    
        if (img[k, middle_width, 0] < 1 - ep) or \
                (img[k, middle_width, 1] < 1 - ep) or \
                (img[k, middle_width, 2] < 1 - ep):
                    found_top = True
     
        if ((img[k, middle_width, 0] > 1 - ep) or \
                (img[k, middle_width, 1] > 1 - ep) or \
                (img[k, middle_width, 2] > 1 - ep)) and \
                found_top:
                    top_coord = k
                    break
    
    ############## Find bottom of frame ###############
    
    found_bot = False
    
    for l in reversed(range(height)):
    
        if (img[l, middle_width, 0] < 1 - ep) or \
                (img[l, middle_width, 1] < 1 - ep) or \
                (img[l, middle_width, 2] < 1 - ep):
                    found_bot = True
     
        if ((img[l, middle_width, 0] > 1 - ep) or \
                (img[l, middle_width, 1] > 1 - ep) or \
                (img[l, middle_width, 2] > 1 - ep)) and \
                found_bot:
                    bot_coord = l
                    break
    
    middle_height_coord = int(round((top_coord + bot_coord)/2.))
    
    ############ Find left of frame ##################
    
    left_coord = width
    
    for n in range(-10, 10):
    
        found_left = False
    
        for k in range(width):
    
            if (img[middle_height_coord + n, k, 0] < 1 - ep) or \
                    (img[middle_height_coord + n, k, 1] < 1 - ep) or \
                    (img[middle_height_coord + n, k, 2] < 1 - ep):
                        found_left = True
     
            if ((img[middle_height_coord + n, k, 0] > 1 - ep) or \
                    (img[middle_height_coord + n, k, 1] > 1 - ep) or \
                    (img[middle_height_coord + n, k, 2] > 1 - ep)) and \
                    found_left:
                        if left_coord > k:
                            left_coord = k
                        break
    
    ############ Find right of frame ##############
    
    right_coord = width
    
    for m in range(-10, 10):
    
        found_right = False
        
        for l in reversed(range(width)):
    
            if (img[middle_height_coord + m, l, 0] < 1 - ep) or \
                    (img[middle_height_coord + m, l, 1] < 1 - ep) or \
                    (img[middle_height_coord + m, l, 2] < 1 - ep):
                        found_right = True
     
            if ((img[middle_height_coord + m, l, 0] > 1 - ep) or \
                    (img[middle_height_coord + m, l, 1] > 1 - ep) or \
                    (img[middle_height_coord + m, l, 2] > 1 - ep)) and \
                    found_right:
                        if right_coord > l:
                            right_coord = l
                        break
    
    middle_width_coord = int(round((right_coord + left_coord)/2.))

    frame_coords = [top_coord, bot_coord, left_coord, right_coord]
    
    return frame_coords


