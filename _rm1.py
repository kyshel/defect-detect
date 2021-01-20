import cv2
from PIL import Image









from time import sleep
from tqdm import tqdm
pbar = tqdm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()

exit()
import os
a = os.path.splitext( '197_10_t20201119085402164_CAM1.jpg__2125__736__512__512__5__0.5019335937499996__0.50095703125__0.017578125__0.015625.jpg')

print(a)

exit()

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