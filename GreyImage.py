import cv2
import numpy as np
img = cv2.imread("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Lena.jpg")
img0 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imwrite("C:/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/grey.jpg",img0)


# 不用 cv 的函数

r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
cv2.imshow("gray", np.uint8(gray))
cv2.waitKey(0)
