from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


driver = webdriver.Firefox()

#driver.get('https://www.svd.se/bors/indexlist.php')

driver.get('https://chart.millistream.com/html5_180402/simple_svd.php?id=6485')

x = [] #np.linspace(0,100,100)
y = [] #np.zeros(100)
h = 1.0

data = driver.find_element_by_xpath('//*[@id="chart"]')

time.sleep(5)
driver.get_screenshot_as_file('/Coding/Python/RealTimePlot/data.png')
driver.close()

################### Find previous Data #####################

img = mpimg.imread('data.png')
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
    for m in range(top_coord + 50, bot_coord + 1):
        
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

# Remove future data

data_pixel = [k for k in data_pixel if k < bot_coord - 1 and k > 0 ]

#Convert pixel values to prices

height_frame = bot_coord - top_coord
data_pixel = [height_frame - k for k in data_pixel]
data = np.zeros(np.shape(data_pixel))

time.sleep(1)

############## Prepare data for plotting ###################

driver = webdriver.Firefox()

driver.get('https://www.svd.se/bors/indexlist.php')

#h = 1.0
x = []
y = []
z = []

dataNew = driver.find_element_by_xpath('/html/body/div[8]/div[2]/table/tbody/tr[2]/td[5]')
dataUpper = driver.find_element_by_xpath('/html/body/div[8]/div[2]/table/tbody/tr[2]/td[6]')
dataLower = driver.find_element_by_xpath('/html/body/div[8]/div[2]/table/tbody/tr[2]/td[7]')

d = dataNew.text.replace(',', '.')
d = d.replace(' ', '')
d = float(d)

lower = dataLower.text.replace(',','.')
lower = lower.replace(' ', '')
upper = dataUpper.text.replace(',','.')
upper = upper.replace(' ', '')

upper_lim = float(upper)
lower_lim = float(lower)

upper_pixel = max(data_pixel)
lower_pixel = min(data_pixel)

if np.argmin(data_pixel) == 0:
    
    print('min')

    pixel_val_differ = (upper_lim - d) / float(upper_pixel - data_pixel[-1])

    for p in range(np.shape(data)[0]):

        data[p] = (d - upper_pixel) * pixel_val_differ + upper_lim

elif np.argmax(data_pixel) == 0:
    
    print('max')

    pixel_val_differ = (d - lower_lim) / float(data_pixel[-1] - lower_pixel)
    
    for p in range(np.shape(data)[0]):

        data[p] = (data_pixel[p] - lower_pixel) * pixel_val_differ + lower_lim

else:
    print('none')
    pixel_val_differ = (upper_lim - lower_lim) / float(upper_pixel - lower_pixel)

    for p in range(np.shape(data)[0]):

        data[p] = (data_pixel[p] - lower_pixel) * pixel_val_differ + lower_lim

localtime = time.localtime(time.time())

################## Prepare plot ###############################


x = np.linspace(9.0, localtime.tm_hour + localtime.tm_min/60.0 + localtime.tm_sec/3600.0 - 15/60., np.shape(data)[0])
y = data
x = np.append(x, localtime.tm_hour + localtime.tm_min/60.0 + localtime.tm_sec/3600.0 - 15/60.)
y = np.append(y, d)


################ Moving average ##############################

long_average = 100
short_average = 50

start_val = 100.0
balance = start_val
stocks = 0.0
money = True
total_net_worth = np.zeros(len(x))
total_net_worth[:long_average] = start_val

for k in range(long_average, len(x)):
    
    average_longterm = np.mean(y[k-long_average:k])
    average_shortterm = np.mean(y[k-short_average:k])

    if average_longterm > average_shortterm and not money:
        # Sell stocks
        balance = stocks*y[k]
        stocks = 0.0
        money = True
    elif average_longterm < average_shortterm and money:
        # Buy stocks
        stocks = balance/y[k]
        balance = 0.0
        money = False

    total_net_worth[k] = balance + stocks*y[k]

plt.ion()

fig = plt.figure(figsize = (16, 8), frameon = None)

ax = fig.add_subplot(2,1,1)
plt.axis([9.0, 17.5, 0.999*min(y), 1.001*max(y)])
bx = fig.add_subplot(2,1,2)
plt.axis([9.0, 17.5, 95, 105])
line1, = ax.plot(x, y, 'b-')
line2, = bx.plot(x, total_net_worth, 'r-')

################ Plot data in real time ######################


print(total_net_worth[:100])
print(total_net_worth[100:150])
print(total_net_worth[150:200])
print(total_net_worth[200:250])

while x[-1] < 17.5:
    
    driver.refresh()
    data = driver.find_element_by_xpath('/html/body/div[8]/div[2]/table/tbody/tr[2]/td[5]')
    d = data.text.replace(',', '.')
    d = d.replace(' ', '')
    
    if float(d) is not y[-1]:
        localtime = time.localtime(time.time())
        x = np.append(x, localtime.tm_hour + localtime.tm_min/60.0 + localtime.tm_sec/3600.0 - 15/60.)
        y = np.append(y, float(d))

        line1.set_xdata(x)
        line1.set_ydata(y)
        
        average_longterm = np.mean(y[-long_average:-1])
        verage_shortterm = np.mean(y[-short_average:-1])

        if average_longterm > average_shortterm and not money:
            # Sell stocks
            balance = stocks*y[-1]
            stocks = 0.0
            money = True
        elif average_longterm < average_shortterm and money:
            # Buy stocks
            stocks = balance/y[-1]
            balance = 0.0
            money = False

        total_net_worth = np.append(total_net_worth, balance + stocks*y[-1])
        
        line2.set_xdata(x)
        line2.set_ydata(total_net_worth)
        
        fig.canvas.draw()
        time.sleep(0.5)
        print(y[-1])
        print(total_net_worth[-1])

    time.sleep(1)
