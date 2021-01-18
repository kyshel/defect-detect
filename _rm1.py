import cv2
from PIL import Image




import torch
# x = torch.rand(5, 3)
# print(x)

print(torch.cuda.is_available())





exit()
img = cv2.imread('ds/step1/197_1_t20201119084916148_CAM1.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im_pil = Image.fromarray(img2)
im_pil.show()

roi_dict.iterrows()
roi_list.enumerate(roi_list)