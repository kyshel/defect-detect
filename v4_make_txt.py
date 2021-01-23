#!/usr/bin/env python
# -*- coding: utf-8 -*-
from v4 import *

# step 2 make txt labels for crops
make_txt_from_crops(CROPS_DIR, TXT_DIR)
exit()






# step 4 cut test big img to tiles
# ask_stale_imgs(TILES_DIR)
crop_holes = {
    0: get_tiles([8192, 6000], TILE_SIZE, LAP_SIZE),
    1: get_tiles([4096, 3500], TILE_SIZE, LAP_SIZE)
}
get_tile_objs_from_test(TEST_DIR, crop_holes)
exit()








# step 1 cut flaw crops
ask_stale_imgs(CROPS_DIR)
get_crop_objs_from_train(TRAIN_DIR, JSON_PATH)  # write to disk each 100 imgs, to prevent men boom
exit()























# make label txt from crops filename

def make_txt_from_crops(dir_crops,dir_txt):
    print('[]make_txt_from_crops')
    # clean dir
    for item in os.listdir(dir_txt):
        if item.endswith(".txt"):
            os.remove(os.path.join(dir_txt, item))

    txt_dict = {}
    for filename in os.listdir(dir_crops):
        if filename.endswith(".jpg"):
            txt_filename = os.path.splitext(filename)[0]
            ele = txt_filename.split('__')
            txt_line = [ele[5],ele[6],ele[7],ele[8],ele[9]]
            txt_dict[txt_filename] = txt_line

    cnt=0
    for filename,line in txt_dict.items():
        with open(dir_txt+filename + ".txt", "w") as text_file:
            text_file.write("{} {} {} {} {}\n".format(line[0],line[1],line[2],line[3],line[4]))
        cnt+=1
        print('txt files cnt:'+str(cnt))

    print('making done,check  '+dir_txt)


dir_crops = CROPS_DIR
dir_txt = TXT_DIR
make_txt_from_crops(dir_crops, dir_txt)





