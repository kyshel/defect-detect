from v5_import import *

# convert bbox(x0,y0,x1,y1) to yolo txt(cx,cy,w,h)
def convert(shape, bbox):
    dw = 1./(shape[0])
    dh = 1./(shape[1])
    x = (bbox[0] + bbox[2])/2.0 - 1
    y = (bbox[1] + bbox[3])/2.0 - 1
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)




