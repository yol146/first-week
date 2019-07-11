import cv2
import matplotlib.pyplot as plt
import numpy as np
# sobel
img = cv2.imread("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Lena.jpg")
x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
Scale_absX = cv2.convertScaleAbs(x)
Scale_absY = cv2.convertScaleAbs(y)
result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
cv2.imwrite("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/sobel_lunkuo.jpg",result)


#leplace

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
cv2.imwrite("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Laplacian_lunkuo.jpg",laplacian)

#Canny
canny = cv2.Canny(img, 50, 100)
cv2.imwrite("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/canny_lunkuo.jpg",canny)
