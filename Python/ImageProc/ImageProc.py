import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('google2.png')
img = img[:,:,:3]
height, width = np.shape(img[:,:,0])

middle_height = round(height/2.0)
middle_width = round(width/2.0)

# Find top of frame
ep = 0.05
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

# Find bottom of frame
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

#img[bot_coord, :, 0] = np.zeros(width) + 0.7
#img[bot_coord, :, 1] = np.zeros(width) + 0.7
#img[bot_coord, :, 2] = np.zeros(width) + 0.7

#img[top_coord, :, 0] = np.zeros(width) + 0.7
#img[top_coord, :, 1] = np.zeros(width) + 0.7
#img[top_coord, :, 2] = np.zeros(width) + 0.7


middle_height_coord = int(round((top_coord + bot_coord)/2.))

# Find top of frame
ep = 0.05
found_left = False

for k in range(width):

    if (img[middle_height_coord, k, 0] < 1 - ep) or \
            (img[middle_height_coord, k, 1] < 1 - ep) or \
            (img[middle_height_coord, k, 2] < 1 - ep):
                found_left = True
 
    if ((img[middle_height_coord, k, 0] > 1 - ep) or \
            (img[middle_height_coord, k, 1] > 1 - ep) or \
            (img[middle_height_coord, k, 2] > 1 - ep)) and \
            found_left:
                left_coord = k
                break

# Find bottom of frame
found_right = False

for l in reversed(range(width)):

    if (img[middle_height_coord, l, 0] < 1 - ep) or \
            (img[middle_height_coord, l, 1] < 1 - ep) or \
            (img[middle_height_coord, l, 2] < 1 - ep):
                found_right = True
 
    if ((img[middle_height_coord, l, 0] > 1 - ep) or \
            (img[middle_height_coord, l, 1] > 1 - ep) or \
            (img[middle_height_coord, l, 2] > 1 - ep)) and \
            found_right:
                right_coord = l
                break

middle_width_coord = int(round((right_coord + left_coord)/2.))

# Extract data
find_data_iterator = 0
found_data = False
data_pixel = np.zeros(right_coord - left_coord)
finished = False

for n in range(left_coord + 1, right_coord + 1):
    iterator = 0
    for m in range(top_coord + 40, bot_coord + 1):
        
        if (img[m, n, 0] < 0.65) or \
                (img[m, n, 1] < 0.45) or \
                (img[m, n, 2] < 0.80):
                    iterator = iterator + 1
        else:
            iterator = 0

        if iterator > 0:
            data_pixel[n - (left_coord + 1)] = m
            iterator = 0
            break

data_pixel = [k for k in data_pixel if k < bot_coord - 1 and k > 0 ]

# Fix data on grid

#print('data_pixel = ', data_pixel[:50])
#print(img[top_coord + 20:top_coord + 70, left_coord + 68, 0])
#print(img[top_coord + 20:top_coord + 70, left_coord + 68, 1])
#print(img[top_coord + 20:top_coord + 70, left_coord + 68, 2])

#Convert pixel values to prices

height_frame = bot_coord - top_coord
data_pixel = [height_frame - k for k in data_pixel]
data = np.zeros(np.shape(data_pixel))

upper_lim = 1512.0
lower_lim = 1487.0
pixel_val_differ = (upper_lim - lower_lim)/height_frame

for p in range(np.shape(data)[0]):

    data[p] = lower_lim + data_pixel[p]*pixel_val_differ


plt.figure(1)
plt.plot(data)
plt.show()

'''
img[top_coord + 200:top_coord + 300, middle_width_coord, 0] = np.zeros(100)
img[top_coord + 200:top_coord + 300, middle_width_coord, 1] = np.zeros(100)
img[top_coord + 200:top_coord + 300, middle_width_coord, 2] = np.zeros(100)

print(img[top_coord + 250:top_coord + 350, middle_width_coord, 0])
print(img[top_coord + 250:top_coord + 350, middle_width_coord, 1])
print(img[top_coord + 250:top_coord + 350, middle_width_coord, 2])
'''

plt.figure(2)
plt.imshow(img[:,:,:3])
plt.show()


