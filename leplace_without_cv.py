from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot
import pylab
import cv2
img0 =  cv2.imread("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Lena.jpg")
img = cv2.cvtColor(img0,cv2.COLOR_RGB2GRAY)
width,height = img.shape
new_image = np.zeros((width, height))
L_sunnzi = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
for x in range(1,(width-1)):
    for y in range(1,(height-1)):
        data =[[img[x-1, y-1], img[x-1, y], img[x-1, y+1]],\
        [img[x, y-1], img[x, y], img[x, y+1]], \
        [img[x+1, y-1], img[x+1, y], img[x+1, y+1]]]
        data= np.array(data)
        var = sum(sum(data * L_sunnzi))
        new_image[x, y] = abs(var)
cv2.imshow("new_image",np.uint8(new_image))
cv2.waitKey(0)
