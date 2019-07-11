from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot
import pylab
import cv2
import math
img0 =  cv2.imread("/Users/N501JW/SummerIntern2019/AIIntern2019/AIIntern2019 1st week/appendix/Lena.jpg")
img = cv2.cvtColor(img0,cv2.COLOR_RGB2GRAY)

# 高斯滤波降噪
img = cv2.GaussianBlur(img,(5,5),0)


# 计算 Grading 以及大小和方向, 并建立相应的矩阵
width,height = img.shape
dx = np.zeros([width-1, height-1])
dy = np.zeros([width-1, height-1])
magnitude = np.zeros([width-1, height-1])
angle = np.zeros([width-1, height-1])
for i in range(width-2):
    for j in range(height-2):
        dx[i, j] = int(img[i+1, j]) - int(img[i, j])
        dy[i, j] = int(img[i, j+1]) - int(img[i, j])
        magnitude[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
        if dy[i,j] == 0:
            angle[i,j] = 0
        else:
            angle[i, j] = math.atan(dx[i, j] /dy[i, j])


# 非极大值的地方进行抑制
NMS = np.zeros([width-1, height-1])
for x in range(1,width-2):
    for y in range(1,height-2):

        # 如果梯度为0，该点就不是边缘点
        if magnitude[x, y] == 0:
            NMS[x, y] = 0
        else:
            gradX = dx[x, y]
            gradY = dy[x, y]
            gradTemp = magnitude[x, y] # 当前梯度点

            # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
            if np.abs(gradY) > np.abs(gradX):
                weight = np.abs(gradX) / np.abs(gradY) # 权重
                grad2 = magnitude[x-1, y]
                grad4 = magnitude[x+1, y]
                # 如果 x, y 方向导数符号一致
                if gradX * gradY > 0:
                    grad1 = magnitude[x-1, y-1]
                    grad3 = magnitude[x+1, y+1]
                else:
                    grad1 = magnitude[x-1, y+1]
                    grad3 = magnitude[x+1, y-1]
            # 如果 x 方向梯度值比较大
            else:
                weight = np.abs(gradY) / np.abs(gradX)
                grad2 = magnitude[x, y-1]
                grad4 = magnitude[x, y+1]
                # 如果 x, y 方向导数符号一致
                if gradX * gradY > 0:
                    grad1 = magnitude[x+1, y-1]
                    grad3 = magnitude[x-1, y+1]
                # 如果 x，y 方向导数符号相反
                else:
                    grad1 = magnitude[x-1, y-1]
                    grad3 = magnitude[x+1, y+1]
            # 利用 grad1-grad4 对梯度进行插值
            gradTemp1 = weight * grad1 + (1 - weight) * grad2
            gradTemp2 = weight * grad3 + (1 - weight) * grad4
            # 当前像素的梯度是局部的最大值，可能是边缘点
            if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                NMS[x, y] = gradTemp
            else:
                # 不可能是边缘点
                NMS[x, y] = 0

W, H = NMS.shape
result = np.zeros([W, H])

# 定义高低阈值
min = 0.1 * np.max(NMS)
max = 0.3 * np.max(NMS)

for i in range(1, W-1):
    for j in range(1, H-1):
       # 双阈值选取
        if (NMS[i, j] < min):
            result[i, j] = 0
        elif (NMS[i, j] > max):
            result[i, j] = 1
        elif (NMS[i-1, j-1:j+1] < max).any() or (NMS[i+1, j-1:j+1].any()
                or (NMS[i, [j-1, j+1]] < max).any()):
            result[i, j] = 1
cv2.imshow("final", result)
cv2.waitKey(0)
