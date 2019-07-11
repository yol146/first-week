from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot
import pylab
import cv2
img0 =  cv2.imread("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Lena.jpg")
img = cv2.cvtColor(img0,cv2.COLOR_RGB2GRAY)
width,height = img.shape
new_image = np.zeros((width, height))
sobel_x =[[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
sobel_y =[[-1, -2, 1],[0, 0, 0],[1, 2, -1]]
for x in range(1,(width-1)):
    for y in range(1,(height-1)):
        data =[[img[x-1, y-1], img[x-1, y], img[x-1, y+1]],\
        [img[x, y-1], img[x, y], img[x, y+1]], \
        [img[x+1, y-1], img[x+1, y], img[x+1, y+1]]]
        data= np.array(data)
        roberts_x = np.array(sobel_x)
        roberts_y = np.array(sobel_y)
        var_x =sum(sum(data * sobel_x))
        var_y = sum(sum(data * sobel_y))
        # combine x and y direction gradient
        var = abs(var_x) + abs(var_y)
        # fill the new_image matrix 
        new_image[x][y] = var
cv2.imshow("new_image",np.uint8(new_image))
cv2.waitKey(0)
