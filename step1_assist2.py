#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_lib1 import *

img_path = r'D:\ml\p8\ds\step1\sample10\hang\197_1_t20201119084916148_CAM1.jpg'

#   ---------------

import cv2 as cv

img_origin = cv.imread(img_path)
img_gray = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)

img = img_gray
img = cv.medianBlur(img,5)


ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
# blur = cv.GaussianBlur(img,(5,5),0)
# ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
thresh = th2







contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_KCOS )
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1 )
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE )
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE  )

print(contours)
print(len(contours[0]))


img = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
img[:] = (255, 255, 255)
cv2.rectangle(img, (0, 0, img.shape[1],img.shape[0]), GREEN, 100)
cv.drawContours(img, contours, -1, RED, 1)







rect = cv2.minAreaRect(contours[0])
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,YELLOW,2)

print(rect)




see(img)

exit()





# ret,thresh = cv.threshold(img,127,255,0)






contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv.moments(cnt)
print( M )



cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])


area = cv.contourArea(cnt)
cv2.imshow('area', area)

cv2.waitKey(0)
cv2.destroyAllWindows()



perimeter = cv.arcLength(cnt,True)

epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)



# hull = cv.convexHull(points[, hull[, clockwise[, returnPoints]]



hull = cv.convexHull(cnt)



k = cv.isContourConvex(cnt)

x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
cv.drawContours(img,[box],0,(0,0,255),2)

(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv.circle(img,center,radius,(0,255,0),2)


ellipse = cv.fitEllipse(cnt)
cv.ellipse(img,ellipse,(0,255,0),2)


rows,cols = img.shape[:2]
[vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)





