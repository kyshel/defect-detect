#!/usr/bin/env python
# -*- coding: utf-8 -*-
from step1_lib1 import *


img_path = r'D:\ml\p8\ds\step1\sample10\hang\197_1_t20201119084916148_CAM1.jpg'
img = cv2.imread(img_path)



#   ---------------































import cv2 as cv



img = cv.imread(img_path,0)


img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)

th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)


titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()



exit()
thresh = th1


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





